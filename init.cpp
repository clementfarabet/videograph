#include "TH.h"
#include "luaT.h"

#include "stdint.h"
#include "set.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define videograph_(NAME) TH_CONCAT_3(videograph_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;

#include "generic/videograph.c"
#include "THGenerateFloatTypes.h"

extern "C" {
  DLL_EXPORT int luaopen_libvideograph(lua_State *L)
  {
    torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
    torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

    videograph_FloatInit(L);
    videograph_DoubleInit(L);

    return 1;
  }
}
