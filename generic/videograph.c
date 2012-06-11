#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/videograph.c"
#else

#ifdef square
#undef square
#endif
#define square(x) ((x)*(x))

#ifdef min
#undef min
#endif
#define min(x,y) (x)<(y) ? (x) : (y)

#ifdef max
#undef max
#endif
#define max(x,y) (x)>(y) ? (x) : (y)

#ifdef rand0to1
#undef rand0to1
#endif
#define rand0to1() ((float)rand()/(float)RAND_MAX)

#ifdef epsilon
#undef epsilon
#endif
#define epsilon 1e-8

static inline real videograph_(ndiff)(real *img,
                                   int nfeats, int height, int width,
                                   int x1, int y1, int z1, int x2, int y2, int z2, char dt) {
  real dist  = 0;
  real dot   = 0;
  real normx = 0;
  real normy = 0;
  real res = 0;
  int i;
  for (i=0; i<nfeats; i++) {
    if (dt == 'e') {
      dist  += square( img[((z1*nfeats+i)*height+y1)*width+x1] - img[((z2*nfeats+i)*height+y2)*width+x2] );
    } else if (dt == 'm') {
      real tmp = fabs( img[((z1*nfeats+i)*height+y1)*width+x1] - img[((z2*nfeats+i)*height+y2)*width+x2] );
      if (tmp > dist) {
        dist = tmp;
      }
    } else if (dt == 'a') {
      dot   += img[((z1*nfeats+i)*height+y1)*width+x1] * img[((z2*nfeats+i)*height+y2)*width+x2];
      normx += square(img[((z1*nfeats+i)*height+y1)*width+x1]);
      normy += square(img[((z2*nfeats+i)*height+y2)*width+x2]);
    }
  }
  if (dt == 'e') res = sqrt(dist);
  else if (dt == 'a') res = acos(dot/(sqrt(normx)*sqrt(normy) + epsilon));
  else if (dt == 'm') res = dist;
  return res;
}

static int videograph_(graph)(lua_State *L) {
  // get args
  THTensor *dst = (THTensor *)luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *src = (THTensor *)luaT_checkudata(L, 2, torch_(Tensor_id));
  int connex = lua_tonumber(L, 3);
  const char *dist = lua_tostring(L, 4);
  char dt = dist[0];

  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);

  // compute all edge weights
  if (connex == 6) {

    // get input dims
    long length, channels, height, width;
    if (src->nDimension == 4) {
      length = src->size[0];
      channels = src->size[1];
      height = src->size[2];
      width = src->size[3];
    } else if (src->nDimension == 3) {
      channels = 1;
      length = src->size[0];
      height = src->size[1];
      width = src->size[2];
    }

    // resize output, and fill it with -1 (which means non-valid edge)
    THTensor_(resize4d)(dst, length, 3, height, width);
    THTensor_(fill)(dst, 0);

    // get raw pointers
    real *src_data = THTensor_(data)(src);
    real *dst_data = THTensor_(data)(dst);

    // build graph with 6-connexity
    long num = 0;
    long x,y,z;
    for (z = 0; z < length; z++) {
      for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
          if (x < width-1) {
            dst_data[((z*3+0)*height+y)*width+x] = videograph_(ndiff)(src_data, channels, height, width,
                                                                           x, y, z, x+1, y, z, dt);
            num++;
          }
          if (y < height-1) {
            dst_data[((z*3+1)*height+y)*width+x] = videograph_(ndiff)(src_data, channels, height, width,
                                                                           x, y, z, x, y+1, z, dt);
            num++;
          }
          if (z < length-1) {
            dst_data[((z*3+2)*height+y)*width+x] = videograph_(ndiff)(src_data, channels, height, width,
                                                                           x, y, z, x, y, z+1, dt);
            num++;
          }
        }
      }
    }

  }

  // cleanup
  THTensor_(free)(src);

  return 0;
}

#ifndef _EDGE_STRUCT_
#define _EDGE_STRUCT_
typedef struct {
  float w;
  int a, b;
} Edge;

void sort_edges(Edge *data, int N)
{
  int i, j;
  real v;
  Edge t;

  if(N<=1) return;

  // Partition elements
  v = data[0].w;
  i = 0;
  j = N;
  for(;;)
    {
      while(data[++i].w < v && i < N) { }
      while(data[--j].w > v) { }
      if(i >= j) break;
      t = data[i]; data[i] = data[j]; data[j] = t;
    }
  t = data[i-1]; data[i-1] = data[0]; data[0] = t;
  sort_edges(data, i-1);
  sort_edges(data+i, N-i);
}
#endif

static int videograph_(segmentmst)(lua_State *L) {
  // get args
  THTensor *dst = (THTensor *)luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *src = (THTensor *)luaT_checkudata(L, 2, torch_(Tensor_id));
  real thres = lua_tonumber(L, 3);
  int minsize = lua_tonumber(L, 4);
  int color = lua_toboolean(L, 5);

  // dims
  long length = src->size[0];
  long nmaps = src->size[1];
  long height = src->size[2];
  long width = src->size[3];

  // make sure input is contiguous
  src = THTensor_(newContiguous)(src);
  real *src_data = THTensor_(data)(src);

  // create edge list from graph (src)
  Edge *edges = NULL; int nedges = 0;
  edges = (Edge *)calloc(length*width*height*nmaps, sizeof(Edge));
  int x,y,z;
  for (z = 0; z < length; z++) {
    for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
        if (x < width-1) {
          edges[nedges].a = (z*height+y)*width+x;
          edges[nedges].b = (z*height+y)*width+(x+1);
          edges[nedges].w = src_data[((z*3+0)*height+y)*width+x];
          nedges++;
        }
        if (y < height-1) {
          edges[nedges].a = (z*height+y)*width+x;
          edges[nedges].b = (z*height+(y+1))*width+x;
          edges[nedges].w = src_data[((z*3+1)*height+y)*width+x];
          nedges++;
        }
        if (z < length-1) {
          edges[nedges].a = (z*height+y)*width+x;
          edges[nedges].b = ((z+1)*height+y)*width+x;
          edges[nedges].w = src_data[((z*3+2)*height+y)*width+x];
          nedges++;
        }
      }
    }
  }

  // sort edges by weight
  sort_edges(edges, nedges);

  // make a disjoint-set forest
  Set *set = set_new(width*height*length);

  // init thresholds
  real *threshold = (real *)calloc(width*height*length, sizeof(real));
  int i;
  for (i = 0; i < width*height*length; i++) threshold[i] = thres;

  // for each edge, in non-decreasing weight order,
  // decide to merge or not, depending on current threshold
  for (i = 0; i < nedges; i++) {
    // components conected by this edge
    int a = set_find(set, edges[i].a);
    int b = set_find(set, edges[i].b);
    if (a != b) {
      if ((edges[i].w <= threshold[a]) && (edges[i].w <= threshold[b])) {
        set_join(set, a, b);
        a = set_find(set, a);
        threshold[a] = edges[i].w + thres/set->elts[a].surface;
      }
    }
  }

  // post process small components
  for (i = 0; i < nedges; i++) {
    int a = set_find(set, edges[i].a);
    int b = set_find(set, edges[i].b);
    if ((a != b) && ((set->elts[a].surface < minsize) || (set->elts[b].surface < minsize)))
      set_join(set, a, b);
  }

  // generate output
  if (color) {
    THTensor *colormap = THTensor_(newWithSize2d)(width*height*length, 3);
    THTensor_(fill)(colormap, -1);
    THTensor_(resize4d)(dst, length, 3, height, width);
    for (z = 0; z < length; z++) {
      for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
          int comp = set_find(set, (z * height + y) * width + x);
          real check = THTensor_(get2d)(colormap, comp, 0);
          if (check == -1) {
            THTensor_(set2d)(colormap, comp, 0, rand0to1());
            THTensor_(set2d)(colormap, comp, 1, rand0to1());
            THTensor_(set2d)(colormap, comp, 2, rand0to1());
          }
          real r = THTensor_(get2d)(colormap, comp, 0);
          real g = THTensor_(get2d)(colormap, comp, 1);
          real b = THTensor_(get2d)(colormap, comp, 2);
          THTensor_(set4d)(dst, z, 0, y, x, r);
          THTensor_(set4d)(dst, z, 1, y, x, g);
          THTensor_(set4d)(dst, z, 2, y, x, b);
        }
      }
    }
  } else {
    THTensor_(resize3d)(dst, length, height, width);
    real *dst_data = THTensor_(data)(dst);
    for (z = 0; z < length; z++) {
      for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
          dst_data[(z*height+y)*width+x] = set_find(set, (z * height + y) * width + x);
        }
      }
    }
  }

  // push number of components
  lua_pushnumber(L, set->nelts);

  // cleanup
  set_free(set);
  free(edges);
  free(threshold);
  THTensor_(free)(src);

  // return
  return 1;
}

int videograph_(colorize)(lua_State *L) {
  // get args
  THTensor *output = (THTensor *)luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *input = (THTensor *)luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *colormap = (THTensor *)luaT_checkudata(L, 3, torch_(Tensor_id));

  // dims
  long length = input->size[0];
  long height = input->size[1];
  long width = input->size[2];

  // generate color map if not given
  if (THTensor_(nElement)(colormap) == 0) {
    THTensor_(resize2d)(colormap, width*height*length, 3);
    THTensor_(fill)(colormap, -1);
  }

  // colormap channels
  int channels = colormap->size[1];

  // generate output
  THTensor_(resize4d)(output, length, channels, height, width);
  int x,y,k,z;
  for (z = 0; z < length; z++) {  
    for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
        int id = THTensor_(get3d)(input, z, y, x);
        real check = THTensor_(get2d)(colormap, id, 0);
        if (check == -1) {
          for (k = 0; k < channels; k++) {
            THTensor_(set2d)(colormap, id, k, rand0to1());
          }
        }
        for (k = 0; k < channels; k++) {
          real color = THTensor_(get2d)(colormap, id, k);
          THTensor_(set4d)(output, z, k, y, x, color);
        }
      }
    }
  }

  // return nothing
  return 0;
}

#ifndef __setneighbor__
#define __setneighbor__
static inline void setneighbor(lua_State *L, long matrix, long id, long idn) {
  // retrieve or create table at index 'id'
  lua_rawgeti(L, matrix, id);
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_createtable(L, 32, 32); // pre-alloc for 32 neighbors
  }
  // append idn
  lua_pushboolean(L, 1);
  lua_rawseti(L, -2, idn);
  // write table back
  lua_rawseti(L, matrix, id);
}
#endif

int videograph_(adjacency)(lua_State *L) {
  // get args
  THTensor *input = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_(Tensor_id)));
  long matrix = 2;

  // dims
  long height = input->size[0];
  long width = input->size[1];

  // raw pointers
  real *input_data = THTensor_(data)(input);

  // generate output
  int x,y;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      long id = input_data[width*y+x];
      if (x < (width-1)) {
        long id_east = input_data[width*y+x+1];
        if (id != id_east) {
          setneighbor(L, matrix, id, id_east);
          setneighbor(L, matrix, id_east, id);
        }
      }
      if (y < (height-1)) {
        long id_south = input_data[width*(y+1)+x];
        if (id != id_south) {
          setneighbor(L, matrix, id, id_south);
          setneighbor(L, matrix, id_south, id);
        }
      }
    }
  }

  // cleanup
  THTensor_(free)(input);

  // return matrix
  return 1;
}

int videograph_(segm2components)(lua_State *L) {
  // get args
  THTensor *segm = (THTensor *)luaT_checkudata(L, 1, torch_(Tensor_id));
  real *segm_data = THTensor_(data)(segm);

  // check dims
  if ((segm->nDimension != 2))
    THError("<videograph.segm2components> segm must be HxW");

  // get dims
  int height = segm->size[0];
  int width = segm->size[1];

  // (0) create a hash table to store all components
  lua_newtable(L);
  int table_hash = lua_gettop(L);

  // (1) get components' info
  long x,y;
  for (y=0; y<height; y++) {
    for (x=0; x<width; x++) {
      // get component ID
      int segm_id = segm_data[width*y+x];

      // get geometry entry
      lua_pushinteger(L,segm_id);
      lua_rawget(L,table_hash);
      if (lua_isnil(L,-1)) {
        // g[segm_id] = nil
        lua_pop(L,1);

        // then create a table to store geometry of component:
        // x,y,size,class,hash
        THTensor *entry = THTensor_(newWithSize1d)(13);
        real *data = THTensor_(data)(entry);
        data[0] = x+1;       // x
        data[1] = y+1;       // y
        data[2] = 1;         // size
        data[3] = 0;         // compat with 'histpooling' method
        data[4] = segm_id;   // hash
        data[5] = x+1;       // left_x
        data[6] = x+1;       // right_x
        data[7] = y+1;       // top_y
        data[8] = y+1;       // bottom_y

        // store entry
        lua_pushinteger(L,segm_id);
        luaT_pushudata(L, entry, torch_(Tensor_id));
        lua_rawset(L,table_hash); // g[segm_id] = entry

      } else {
        // retrieve entry
        THTensor *entry = (THTensor *)luaT_toudata(L, -1, torch_(Tensor_id));
        lua_pop(L,1);

        // update content
        real *data = THTensor_(data)(entry);
        data[0] += x+1;       // x += x + 1
        data[1] += y+1;       // y += y + 1
        data[2] += 1;         // size += 1
        data[5] = (x+1)<data[5] ? x+1 : data[5];   // left_x
        data[6] = (x+1)>data[6] ? x+1 : data[6];   // right_x
        data[7] = (y+1)<data[7] ? y+1 : data[7];   // top_y
        data[8] = (y+1)>data[8] ? y+1 : data[8];   // bottom_y
      }
    }
  }

  // (2) traverse geometry table to produce final component list
  lua_pushnil(L);
  while (lua_next(L, table_hash) != 0) {
    // retrieve entry
    THTensor *entry = (THTensor *)luaT_toudata(L, -1, torch_(Tensor_id)); lua_pop(L,1);
    real *data = THTensor_(data)(entry);

    // normalize cx and cy, by component's size
    long size = data[2];
    data[0] /= size;  // cx/size
    data[1] /= size;  // cy/size

    // extra info
    data[9] = data[6] - data[5] + 1;     // box width
    data[10] = data[8] - data[7] + 1;    // box height
    data[11] = (data[6] + data[5]) / 2;  // box center x
    data[12] = (data[8] + data[7]) / 1;  // box center y
  }

  // return component table
  return 1;
}

static const struct luaL_Reg videograph_(methods__) [] = {
  {"graph", videograph_(graph)},
  {"segmentmst", videograph_(segmentmst)},
  {"colorize", videograph_(colorize)},
  {"adjacency", videograph_(adjacency)},
  {"segm2components", videograph_(segm2components)},
  {NULL, NULL}
};

static void videograph_(Init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, videograph_(methods__), "videograph");
  lua_pop(L,1);
}

#endif