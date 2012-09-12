#include "TH.h"
#include "luaT.h"

#include "stdint.h"
#include "set.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define videograph_(NAME) TH_CONCAT_3(videograph_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/videograph.c"
#include "THGenerateFloatTypes.h"

extern "C" {
  DLL_EXPORT int luaopen_libvideograph(lua_State *L)
  {
    videograph_FloatInit(L);
    videograph_DoubleInit(L);

    return 1;
  }
}
