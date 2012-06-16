----------------------------------------------------------------------
--
-- Copyright (c) 2012 Clement Farabet
--               2006 Pedro Felzenszwalb
--
-- This program is free software; you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation; either version 2 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.
--
-- You should have received a copy of the GNU General Public License
-- along with this program; if not, write to the Free Software
-- Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
--
----------------------------------------------------------------------
-- description:
--     videograph - a graph package for videos:
--                  this package extends imgraph, for videos.
----------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'image'
require 'imgraph'

-- create global package table:
videograph = {}

-- c lib:
require 'libvideograph'

----------------------------------------------------------------------
-- computes a graph from a video (3D or 4D array)
--
function videograph.graph(...)
   -- get args
   local args = {...}
   local dest, video, connex, distance
   local arg2 = torch.typename(args[2])
   if arg2 and arg2:find('Tensor') then
      dest = args[1]
      video = args[2]
      connex = args[3]
      distance = args[4]
      flow = args[5]
   else
      video = args[1]
      connex = args[2]
      distance = args[3]
      flow = args[4]
   end

   -- defaults
   connex = connex or 6
   distance = ((not distance) and 'e') or ((distance == 'euclid') and 'e') 
              or ((distance == 'angle') and 'a') or ((distance == 'max') and 'm') 

   -- usage
   if not video or (connex ~= 6) or (distance ~= 'e' and distance ~= 'a' and distance ~= 'm') then
      print(xlua.usage('videograph.graph',
                       'compute an edge-weighted graph on a video sequence\n'
                       + '(if a flow field is passed, edges are warped through time, accoring to the field;\n'
                       + ' the field should be computed backwards, i.e. from frame (t+1) to frame (t)',
                       nil,
                       {type='torch.Tensor', help='input tensor (for now LxKxHxW or LxHxW)', req=true},
                       {type='number', help='connexity (edges per vertex): 6', default=6},
                       {type='string', help='distance metric: euclid | angle | max', req='euclid'},
                       {type='torch.Tensor', help='optional flow field, to constrain time edges (Lx2xHxW)'},
                       "",
                       {type='torch.Tensor', help='destination: existing graph', req=true},
                       {type='torch.Tensor', help='input tensor (for now LxKxHxW or LxHxW)', req=true},
                       {type='number', help='connexity (edges per vertex): 6', default=6},
                       {type='string', help='distance metric: euclid | angle | max', req='euclid'},
                       {type='torch.Tensor', help='optional flow field, to constrain time edges (Lx2xHxW)'}))
      xlua.error('incorrect arguments', 'videograph.graph')
   end

   -- create dest
   dest = dest or torch.Tensor():typeAs(video)

   -- compute graph
   if flow then
      video.videograph.flowgraph(dest, video, flow, connex, distance)
   else
      video.videograph.graph(dest, video, connex, distance)
   end

   -- return result
   return dest
end

----------------------------------------------------------------------
-- segment a graph, by computing its min-spanning tree and
-- merging vertices based on a dynamic threshold
--
function videograph.segmentmst(...)
   --get args
   local args = {...}
   local dest, graph, thres, minsize, colorize
   local arg2 = torch.typename(args[2])
   if arg2 and arg2:find('Tensor') then
      dest = args[1]
      graph = args[2]
      thres = args[3]
      minsize = args[4]
      colorize = args[5]
   else
      graph = args[1]
      thres = args[2]
      minsize = args[3]
      colorize = args[4]
   end

   -- defaults
   thres = thres or 3
   minsize = minsize or 20
   colorize = colorize or false

   -- usage
   if not graph then
      print(xlua.usage('videograph.segmentmst',
                       'segment an edge-weighted graph, using a surface adaptive criterion\n'
                          .. 'on the min-spanning tree of the graph (see Felzenszwalb et al. 2004)',
                       nil,
                       {type='torch.Tensor', help='input graph', req=true},
                       {type='number', help='base threshold for merging', default=3},
                       {type='number', help='min size: merge components of smaller size', default=20},
                       {type='boolean', help='replace components id by random colors', default=false},
                       "",
                       {type='torch.Tensor', help='destination tensor', req=true},
                       {type='torch.Tensor', help='input graph', req=true},
                       {type='number', help='base threshold for merging', default=3},
                       {type='number', help='min size: merge components of smaller size', default=20},
                       {type='boolean', help='replace components id by random colors', default=false}))
      xlua.error('incorrect arguments', 'videograph.segmentmst')
   end

   -- compute segmented video
   dest = dest or torch.Tensor():typeAs(graph) 
   local nelts
   if graph:nDimension() == 4 then
      -- dense image graph (input is an LxKxHxW graph, L=video length, K=1/2 connexity, nnodes=H*W*L)
      nelts = graph.videograph.segmentmst(dest, graph, thres, minsize, colorize)
   else
      -- sparse graph (input is a Nx3 graph, nnodes=N, each entry input[i] is an edge: {node1, node2, weight})
      nelts = graph.imgraph.segmentmstsparse(dest, graph, thres, minsize, colorize)
   end

   -- return segmented video
   return dest, nelts
end

----------------------------------------------------------------------
-- extract information/geometry of a segmentation's components
--
function videograph.extractcomponents(...)
   -- get args
   local args = {...}
   local input = args[1]
   local video = args[2]
   local config = args[3] or 'bbox'
   local encoder = args[4]
   local minsize = args[5] or 1

   -- usage
   if not input then
      print(
         xlua.usage(
            'videograph.extractcomponents',
            'return a list of structures describing the components of a segmentation. \n'
               .. 'if a KxHxW image is given, then patches can be extracted from it, \n'
               .. 'and appended to the list returned. \n'
               .. 'the optional config string specifies how these patches should be \n'
               .. 'returned (bbox: raw bounding boxes, mask: binary segmentation mask, \n'
               .. 'masked: bbox masked by segmentation mask)',
            'graph = videograph.graph(image.lena())\n'
               .. 'segm = videograph.segmentmst(graph)\n'
               .. 'components = videograph.extractcomponents(segm)',
            {type='torch.Tensor',  help='input segmentation map (must be LxHxW), and each element must be in [1,NCLASSES]', req=true},
            {type='torch.Tensor', help='auxiliary video: if given, then components are cropped from it (must be LxKxHxW)'},
            {type='string', help='configuration, one of: bbox | masked', default='bbox'},
            {type='function', help='encoder: function that encodes cropped/masked patches into a code (doing it here can save a lot of memory)'},
            {type='number', help='minimum component size to process', default=1}
         )
      )
      xlua.error('incorrect arguments', 'videograph.extractcomponents')
   end

   -- support LongTensors
   if torch.typename(input) == 'torch.LongTensor' then
      input = torch.Tensor(input:size(1), input:size(2), input:size(3)):copy(input)
   end

   -- generate lists
   local hcomponents
   local masks = {}
   if torch.typename(input) then
      hcomponents = input.videograph.segm2components(input)
   else
      error('please provide input')
   end

   -- reorganize
   local components = {centroid_x={}, centroid_y={}, centroid_z={}, surface={}, 
                       id = {}, revid = {},
                       bbox_width = {}, bbox_height = {}, bbox_length = {},
                       bbox_top = {}, bbox_bottom = {}, 
                       bbox_left = {}, bbox_right = {},
                       bbox_first = {}, bbox_last = {},
                       bbox_x = {}, bbox_y = {}, bbox_z = {}, patch = {}, mask = {}}
   local i = 0
   for _,comp in pairs(hcomponents) do
      i = i + 1
      components.centroid_x[i]  = comp[1]
      components.centroid_y[i]  = comp[2]
      components.centroid_z[i]  = comp[3]
      components.surface[i]     = comp[4]
      components.id[i]          = comp[6]
      components.revid[comp[6]] = i
      components.bbox_left[i]   = comp[7]
      components.bbox_right[i]  = comp[8]
      components.bbox_top[i]    = comp[9]
      components.bbox_bottom[i] = comp[10]
      components.bbox_first[i]  = comp[11]
      components.bbox_last[i]   = comp[12]
      components.bbox_width[i]  = comp[13]
      components.bbox_height[i] = comp[14]
      components.bbox_length[i] = comp[15]
      components.bbox_x[i]      = comp[16]
      components.bbox_y[i]      = comp[17]
      components.bbox_z[i]      = comp[18]
   end
   components.size = function(self) return #self.surface end

   -- auxiliary video given ?
   if video and video:nDimension() == 4 then
      local c = components
      local maskit = false
      if config == 'masked' then maskit = true end
      for k = 1,i do
         if c.surface[k] >= minsize then
            -- get bounding box corners:
            local top = c.bbox_top[k]
            local bottom = c.bbox_bottom[k]
            local height = c.bbox_height[k]
            
            local left = c.bbox_left[k]
            local right = c.bbox_right[k]
            local width = c.bbox_width[k]
            
            local first = c.bbox_first[k]
            local last = c.bbox_last[k]
            local length = c.bbox_length[k]

            -- extract patch from image:
            c.patch[k] = video[{ {first,last},{},{top,bottom},{left,right} }]:clone()

            -- generate mask, if not available
            if torch.typename(input) and not c.mask[k] then
               -- the input is a grayscale image, crop it to get the mask:
               c.mask[k] = input[{ {first,last},{top,bottom},{left,right} }]:clone()
               local id = components.id[k]
               c.mask[k]:apply(function(x) 
                  if x == id then return 1 else return 0 end 
               end)
            end

            -- mask box
            if maskit then
               for i = 1,c.patch[k]:size(2) do
                  c.patch[k][{ {},i,{},{} }]:cmul(c.mask[k])
               end
            end

            -- encoder?
            if encoder then
               c.descriptor = c.descriptor or {}
               c.descriptor[k] = encoder(c.patch[k], c.mask[k])
               c.patch[k] = nil
               c.mask[k] = nil
               collectgarbage()
            end
         end
      end
   end

   -- return both lists
   return components
end

----------------------------------------------------------------------
-- colorize a segmentation map
--
function videograph.colorize(...)
   -- get args
   local args = {...}
   local grayscale = args[1]
   local colormap = args[2]

   -- usage
   if not grayscale or not (grayscale:dim() == 3 or (grayscale:dim() == 4 and grayscale:size(2) == 1)) then
      print(xlua.usage('videograph.colorize',
                       'colorize a segmentation map',
                       'graph = videograph.graph(image.lena())\n'
                          .. 'segm = videograph.segmentmst(graph)\n'
                          .. 'colored = videograph.colorize(segm)',
                       {type='torch.Tensor', help='input segmentation map (must be HxW), and each element must be in [1,width*height]', req=true},
                       {type='torch.Tensor', help='color map (must be Nx3), if not provided, auto generated'}))
      xlua.error('incorrect arguments', 'videograph.colorize')
   end

   -- accept 4D grayscale
   if grayscale:dim() == 4 and grayscale:size(2) == 1 then
      grayscale = grayscale:new():resize(grayscale:size(1), grayscale:size(3), grayscale:size(4))
   end

   -- support LongTensors
   if torch.typename(grayscale) == 'torch.LongTensor' then
      grayscale = torch.Tensor(grayscale:size(1), grayscale:size(2), grayscale:size(3)):copy(grayscale)
   end

   -- auto type
   colormap = colormap or torch.Tensor():typeAs(grayscale)
   local colorized = torch.Tensor():typeAs(grayscale)

   -- colorize !
   grayscale.videograph.colorize(colorized, grayscale, colormap)

   -- return colorized segmentation
   return colorized, colormap
end

----------------------------------------------------------------------
-- return the adjacency matrix of a segmentation map
--
function videograph.adjacency(...)
   -- get args
   local args = {...}
   local input = args[1]
   local components = args[2]
   local directed = args[3] or false

   -- usage
   if not input then
      print(xlua.usage('videograph.adjacency',
                       'return the adjacency matrix of a segmentation map.\n\n'
                          .. 'a component list can be given, in which case the list\n'
                          .. 'is updated to directly embed the neighboring relationships\n'
                          .. 'and a second adjacency matrix is returned, using the revids\n'
                          .. 'available in the component list',
                       'graph = videograph.graph(image.lena())\n'
                          .. 'segm = videograph.segmentmst(graph)\n'
                          .. 'matrix = videograph.adjacency(segm)\n\n'
                          .. 'components = videograph.extractcomponents(segm)\n'
                          .. 'segm = videograph.adjacency(segm, components)\n'
                          .. 'print(components.neighbors) -- list of neighbor IDs\n'
                          .. 'print(components.adjacency) -- adjacency matrix of IDs',
                       {type='torch.Tensor', help='input segmentation map (must be LxHxW), and each element must be in [1,NCLASSES]', req=true},
                       {type='table', help='component list, as returned by videograph.extractcomponents()'}))
      xlua.error('incorrect arguments', 'videograph.adjacency')
   end

   -- support LongTensors
   if torch.typename(input) and torch.typename(input) == 'torch.LongTensor' then
      input = torch.Tensor(input:size(1), input:size(2)):copy(input)
   end

   -- fill matrix
   local adjacency
   if torch.typename(input) then
      adjacency = input.videograph.adjacency(input, {})
   else
      error('please provide an input')
   end

   -- update component list, if given
   if components then
      components.neighbors = {}
      components.adjacency = {}
      for i = 1,components:size() do
         local neighbors = adjacency[components.id[i]]
         local ntable = {}
         local ktable = {}
         if neighbors then
            for id in pairs(neighbors) do
               table.insert(ntable, components.revid[id])
               ktable[components.revid[id]] = true
            end
         end
         components.neighbors[i] = ntable
         components.adjacency[i] = ktable
      end
   end

   -- return adjacency matrix
   return adjacency
end

----------------------------------------------------------------------
-- test me functions
--
function videograph.testme_simple(path)
   if not path then
      print('please provide path to video file: testme("path/to/video")')
      return
   end
   require 'ffmpeg'
   print '<videograph> loading video'
   video = ffmpeg.Video{path=path, width=500, height=330, fps=10, length=5,
                        encoding='ppm', delete=false}
   print '<videograph> exporting video to tensor'
   input = video:totensor{}
   print '<videograph> smoothing video'
   for i = 1,(#input)[1] do
      input[i] = image.convolve(input[i], image.gaussian(3), 'same')
   end
   print '<videograph> constructing graph'
   graph = videograph.graph(input)
   print '<videograph> segmenting graph'
   segm = videograph.segmentmst(graph,5,200)
   print '<videograph> colorize segmentation'
   segmc = videograph.colorize(segm)
   print '<videograph> creating video from graph'
   processed = ffmpeg.Video{tensor=segmc, fps=10}
   video:play{loop=true}
   processed:play{loop=true}
   print '<videograph> done.'
end

function videograph.testme_flow(path)
   if not path then
      print('please provide path to video file: testme("path/to/video")')
      return
   end
   require 'ffmpeg'
   require 'liuflow'
   torch.setdefaulttensortype('torch.FloatTensor')
   print '<videograph> loading video'
   video = ffmpeg.Video{path=path, width=500, height=330, fps=20, length=1,
                        encoding='ppm', delete=false}
   print '<videograph> exporting video to tensor'
   input = video:totensor{}
   print '<videograph> smoothing video'
   for i = 1,(#input)[1] do
      input[i] = image.convolve(input[i], image.gaussian(3), 'same')
   end
   print '<videograph> computing dense optical flow'
   flow = torch.input( (#input)[1], 2, (#input)[3], (#input)[4] )
   for i = 1,(#input)[1] do
      xlua.progress(i,(#input)[1])
      if i > 1 then
         n,a,_,x,y = liuflow.infer{pair={input[i],input[i-1]},
                                   alpha=1e-2, minWidth=50,
                                   nCGIterations=5, nOuterFPIterations=5}
         flow[i][1] = x
         flow[i][2] = y
         image.display(liuflow.xy2rgb(flow[i][1],flow[i][2]))
      end
   end
   print '<videograph> constructing graph'
   graph = videograph.graph(input,6,'euclid',flow)
   print '<videograph> segmenting graph'
   segm = videograph.segmentmst(graph,1,100)
   print '<videograph> colorize segmentation'
   segm = videograph.colorize(segm)
   print '<videograph> creating video from graph'
   processed = ffmpeg.Video{tensor=segm, fps=20}
   video:play{loop=true}
   processed:play{loop=true}
   print '<videograph> done.'
end

function videograph.testme_adjacency(path)
   -- run basic test
   videograph.testme_simple(path)
   print '<videograph> compute adjacency matrix'
   comps = videograph.extractcomponents(segm,input,'masked')
   videograph.adjacency(segm,comps)
end
