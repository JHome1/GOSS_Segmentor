# -----------------------------------------
# Project: 'GOSS Segmentor' 
# Written by Jie Hong (jie.hong@anu.edu.au)
# -----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

PI = 3.14159265
REPLUSIVE_INIT = torch.tensor([2147483648])
direction = torch.tensor([[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])


def find(parent, x):
    if parent[x] == x: return x
    else: 
        parent[x] = find(parent, parent[x])
        return parent[x]


def FIND(L, index):
    label = L[index]
    
    while not label - 1 == index:
        index = label-1
        label = L[index]

    return index


def UNION(L, a, b): 
    done = False

    while done==False:
       a = FIND(L, a)
       b = FIND(L, b)

       if a==b: done = True
       else: 
          if done==False and a>b:
              tmp = a
              a = b
              b = tmp
          
          L[b] = torch.min(torch.tensor([L[b], a+1]))
          old = L[b]
         
          if old==b+1: done = True 
          b = old - 1  

    return L   


class bpd_cuda_python(nn.Module):
      def __init__(self):
          super(bpd_cuda_python, self).__init__() 

      def find_parents(self, nthreads, height, width, theta_a, input_angles, parents, roots):
          for i in range(height):
              for j in range(width):
                  curr_angle = input_angles[i, j]
                  pos = (curr_angle + PI/8)/(PI/4)
                  pos = torch.round(pos)
                  if (pos >= 8): pos = pos-8
                  pos = pos.long()

                  next_h = i + direction[pos, 0]
                  next_h = next_h.long()
                  next_w = j + direction[pos, 1]
                  next_w = next_w.long()

                  if next_h >= height or next_h <0 or next_w >= width or next_w <0: 
                      parents[0, i, j] = i
                      parents[1, i, j] = j
                      roots[i, j] = 1
                  else: 
                      next_angle = input_angles[next_h][next_w]
                      angle_diff = torch.abs(curr_angle-next_angle) 
                      angle_diff = torch.min(torch.tensor([angle_diff, 2*PI-angle_diff]))

                      if angle_diff > (theta_a*PI/180):
                          parents[0, i, j] = i
                          parents[1, i, j] = j
                          roots[i, j] = 1
                      else:       
                          parents[0, i, j] = next_h
                          parents[1, i, j] = next_w

          return parents, roots

      def get_super_BPDs_step1(self, nthreads, height, width, parents, super_BPDs):
          for i in range(height):
              for j in range(width):
                  next_h = parents[0, i, j].long()
                  next_w = parents[1, i, j].long()                  
                  next_index = next_h*width + next_w

                  super_BPDs = UNION(super_BPDs.detach().cpu(), i*width+j, next_index.detach().cpu())

          return super_BPDs.cuda()

      def get_super_BPDs_step2(self, nthreads, super_BPDs):
          for i in range(nthreads):
              super_BPDs[i] = FIND(super_BPDs.detach().cpu(), i) + 1

          return super_BPDs.cuda()

      def merge_nearby_root_pixels(self, nthreads, height, width, roots, super_BPDs):
          for i in range(height):
              for j in range(width):
                  if roots[i, j]!=0: 
                     for delta_h in range(0, torch.min(torch.tensor([3, height-1-i]))):
                         for delta_w in range(-torch.min(torch.tensor([3, j])), torch.min(torch.tensor([3, width-1-j]))):
                             next_h = i + delta_h
                             next_w = j + delta_w
                             if roots[next_h, next_w]==1:
                                 next_index = next_h*width + next_w
                                 super_BPDs = UNION(super_BPDs.detach().cpu(), i*width+j, next_index)

          return super_BPDs.cuda()

      def find_bnd_angle_diff(self, nthreads, height, width, num_superpixels, input_angles,
                              super_BPDs, parents, unique_super_BPDs_inverse, bnd_angle_diff, bnd_pair_nums):

          for i in range(height):
              for j in range(width): 
                  curr_index = i*width + j
                  # right and bottom point
                  delta_h = [0, 1]
                  delta_w = [1, 0]

                  for k in range(2):
                      next_h = i + delta_h[k]
                      next_w = j + delta_w[k]        

                      if next_w >= width or next_h >= height: continue
                
                      next_index = next_h*width + next_w
                           
                      if not (super_BPDs[curr_index] == super_BPDs[next_index]): 
                          curr_position = unique_super_BPDs_inverse[i, j].long()
                          next_position = unique_super_BPDs_inverse[next_h, next_w].long()

                          min_position = torch.min(torch.tensor([curr_position, next_position])) 
                          max_position = torch.max(torch.tensor([curr_position, next_position])) 

                          bnd_pair_nums[min_position, max_position] = bnd_pair_nums[min_position, max_position] + 1
                          # forward 3 steps respectively, then calculate angle diff
                          curr_h = i
                          curr_w = j
                          for step in range(3):
                              curr_parent_h = parents[0, curr_h, curr_w].long()
                              curr_parent_w = parents[1, curr_h, curr_w].long()
                              curr_h = curr_parent_h
                              curr_w = curr_parent_w
  
                              next_parent_h = parents[0, next_h, next_w].long()
                              next_parent_w = parents[1, next_h, next_w].long()
                              next_h = next_parent_h
                              next_w = next_parent_w
                                          
                          curr_angle = input_angles[curr_h, curr_w].float()
                          next_angle = input_angles[next_h, next_w].float()                   
                          angle_diff = torch.abs(curr_angle-next_angle)
                          angle_diff = torch.min(torch.tensor([angle_diff, 2*PI - angle_diff]))
                          bnd_angle_diff[min_position, max_position] = bnd_angle_diff[min_position, max_position] + angle_diff

          return bnd_angle_diff, bnd_pair_nums 

      def classify_edges(self, nthreads, num_superpixels, nums, S_o, bnd_angle_diff, bnd_pair_nums, select_matrix, edge_h, edge_w, replusive_matrix):
          
          for i in range(num_superpixels):
              for j in range(nums): 
                  if bnd_pair_nums[i, j] != 0:           
                      avg_angle_diff = bnd_angle_diff[i,j] / bnd_pair_nums[i, j] 
                      avg_angle_diff.float()
                      bnd_angle_diff[i, j] = avg_angle_diff

                      if avg_angle_diff > PI-S_o*PI/180:
                          inter_h = j/32
                          inter_h = torch.tensor([inter_h]).long().cuda()
                          inter_w = j%32
                          inter_w = torch.tensor([inter_w]).long().cuda()

                          replusive_matrix[i, inter_h] = (replusive_matrix[i, inter_h]) | (REPLUSIVE_INIT.cuda()>>inter_w)
                      
                      else:
                          select_matrix[i, j] = 1
                          edge_h[i, j] = i
                          edge_w[i, j] = j
 
          return select_matrix, replusive_matrix, edge_h, edge_w         

      def final_step(self, num_edges, connect_marks, edge_h, edge_w, 
                     unique_super_BPDs, super_BPDs):
          for i in range(num_edges):
              if connect_marks[i] == 1:
                  index_h = unique_super_BPDs[edge_h[i]]-1
                  index_h = index_h.long()
                  index_w = unique_super_BPDs[edge_w[i]]-1
                  index_w = index_w.long()

                  super_BPDs = UNION(super_BPDs, index_h, index_w)

          return super_BPDs   

      def bpd_cuda(self, input_angles, height, width, theta_a, S_o):
          # get parents and roots
          parents = torch.zeros([2, height, width]).cuda()
          roots   = torch.zeros([height, width]).cuda() 
          parents, roots = self.find_parents(height*width, height, width, theta_a, input_angles, parents, roots) 
 
          # get super-BPDs
          super_BPDs = torch.arange(1, height*width+1).cuda()
          super_BPDs = self.get_super_BPDs_step1(height*width, height, width, parents, super_BPDs)
          super_BPDs = self.get_super_BPDs_step2(height*width, super_BPDs)

          super_BPDs_before_dilation = super_BPDs
          super_BPDs_before_dilation = super_BPDs_before_dilation.reshape(height, width)
       
          # merge nearby root pixels
          super_BPDs = self.merge_nearby_root_pixels(height*width, height, width, roots, super_BPDs)
          super_BPDs = self.get_super_BPDs_step2(height*width, super_BPDs)

          super_BPDs_after_dilation = super_BPDs
          super_BPDs_after_dilation = super_BPDs_after_dilation.reshape(height, width)

          # contruct RAG
          unique_super_BPDs, unique_super_BPDs_inverse, unique_super_BPDs_counts = torch.unique(super_BPDs, sorted=True, return_inverse=True, return_counts=True)
          unique_super_BPDs_inverse = unique_super_BPDs_inverse.reshape(height, width)          

          num_superpixels = torch.numel(unique_super_BPDs)
          bnd_angle_diff = torch.zeros([num_superpixels, num_superpixels]).cuda()
          bnd_pair_nums  = torch.zeros([num_superpixels, num_superpixels]).cuda()

          bnd_angle_diff, bnd_pair_nums = self.find_bnd_angle_diff(height*width, height, width,
                                                                   num_superpixels, input_angles,
                                                                   super_BPDs, parents, unique_super_BPDs_inverse,
                                                                   bnd_angle_diff, bnd_pair_nums)

          # classify edges
          select_matrix = torch.zeros([num_superpixels, num_superpixels]).cuda()
          edge_h = torch.zeros([num_superpixels, num_superpixels]).long().cuda()
          edge_w = torch.zeros([num_superpixels, num_superpixels]).long().cuda()
          nums = (num_superpixels + 32 - 1) / 32
          nums = torch.round(torch.tensor([nums])).long()
          replusive_matrix = torch.zeros([num_superpixels, nums]).long().cuda()

          select_matrix, replusive_matrix, edge_h, edge_w = self.classify_edges(num_superpixels*num_superpixels, num_superpixels, nums, S_o,
                                                                                bnd_angle_diff, bnd_pair_nums,
                                                                                select_matrix, 
                                                                                edge_h, edge_w,
                                                                                replusive_matrix)

          select_matrix = select_matrix.bool()
          bnd_angle_diff = torch.masked_select(bnd_angle_diff, select_matrix)
          edge_h = torch.masked_select(edge_h, select_matrix)
          edge_w = torch.masked_select(edge_w, select_matrix)

          # diff small to large, sim large to small
          sort_index = torch.argsort(bnd_angle_diff)
          sorted_bnd_angle_diff = bnd_angle_diff[sort_index]
          sorted_edge_h = edge_h[sort_index]
          sorted_edge_w = edge_w[sort_index]
  
          # connect edges
          return unique_super_BPDs_counts, sorted_edge_h, sorted_edge_w, sorted_bnd_angle_diff, replusive_matrix, unique_super_BPDs, roots, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs 

      def bpd_cuda_final_step(self, height, width, connect_marks, edge_h, edge_w, unique_super_BPDs, super_BPDs):
          num_edges = torch.numel(edge_h)

          super_BPDs = self.final_step(num_edges, connect_marks,
                                       edge_h, edge_w, 
                                       unique_super_BPDs, super_BPDs)
          super_BPDs = self.get_super_BPDs_step2(height*width, super_BPDs)              
          super_BPDs = super_BPDs.reshape(height, width)

          return super_BPDs

      def bpd_cpu_forward(self, unique_super_BPDs_counts, edge_h, edge_w, bnd_angle_diff, replusive_matrix, theta_l, theta_s):
          num_edges = torch.numel(edge_h)
          num_superpixels = replusive_matrix.shape[0]
          num_32 = replusive_matrix.shape[1]

          parent = torch.zeros([num_superpixels]).long().cuda()
          for i in range(num_superpixels): 
              parent[i] = i

          connect_marks = torch.zeros([num_edges]).cuda()

          for i in range(num_edges):
              index_h = find(parent, edge_h[i]).long()             
              index_w = find(parent, edge_w[i]).long()

              area_h = unique_super_BPDs_counts[index_h].long()
              area_w = unique_super_BPDs_counts[index_w].long()

              min_area = torch.min(torch.tensor([area_h, area_w])).long()

              thresh = 0
    
              inter_h = index_w/32
              inter_h = inter_h.long()
              inter_w = index_w%32
              inter_w = inter_w.long()
              value = REPLUSIVE_INIT.cuda()>>inter_w

              if (min_area > 250) and (not replusive_matrix[index_h, inter_h]&value):
                  if min_area > 1500: 
                      thres = PI - theta_l*PI/180
                  else: 
                      thres = PI - theta_s*PI/180  
                
                  if bnd_angle_diff[i] < thres:
                      connect_marks[i] = 1
                      parent[index_h] = index_w
                      # update area and replusive matrix
                      for j in range(num_32):
                          replusive_matrix[index_w, j] = replusive_matrix[index_w, j] | replusive_matrix[index_h, j]
                      unique_super_BPDs_counts[index_w] = area_h + area_w

          # tiny region    
          for i in range(num_edges):
              if connect_marks[i]==1: continue    

              index_h = find(parent, edge_h[i]).long()             
              index_w = find(parent, edge_w[i]).long()
              
              area_h = unique_super_BPDs_counts[index_h].long()
              area_w = unique_super_BPDs_counts[index_w].long()

              min_area = torch.min(torch.tensor([area_h, area_w])).long()

              thresh = 0
    
              inter_h = index_w/32
              inter_h = inter_h.long()
              inter_w = index_w%32
              inter_w = inter_w.long()
              value = REPLUSIVE_INIT.cuda()>>inter_w

              if (min_area <= 250) and (not replusive_matrix[index_h, inter_h]&value):
                  connect_marks[i] = 1
                  parent[index_h] = index_w
                  # update area and replusive_matrix
                  for j in range(num_32):
                      replusive_matrix[index_w, j] = replusive_matrix[index_w, j] | replusive_matrix[index_h, j]
                  unique_super_BPDs_counts[index_w] = area_h + area_w  

          return connect_marks

      def forward(self, input_angles, height, width, theta_a, theta_l, theta_s, S_o):
          unique_super_BPDs_counts, edge_h, edge_w, bnd_angle_diff, replusive_matrix, unique_super_BPDs, root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs = self.bpd_cuda(input_angles, height, width, theta_a, S_o)

          connect_marks = self.bpd_cpu_forward(unique_super_BPDs_counts,         \
                                               edge_h, edge_w,                   \
                                               bnd_angle_diff, replusive_matrix, \
                                               theta_l, theta_s)
          final_result = self.bpd_cuda_final_step(height, width, connect_marks, edge_h, edge_w, unique_super_BPDs, super_BPDs)
 
          return root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, final_result 
