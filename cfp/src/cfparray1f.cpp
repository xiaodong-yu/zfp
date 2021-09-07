#include "cfparray1f.h"
#include "zfparray1.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array1f
#define CFP_REF_TYPE cfp_ref1f
#define CFP_PTR_TYPE cfp_ptr1f
#define CFP_ITER_TYPE cfp_iter1f
#define ZFP_ARRAY_TYPE zfp::array1f
#define ZFP_SCALAR_TYPE float

#define CFP_CONTAINER_TYPE CFP_ARRAY_TYPE
#define ZFP_CONTAINER_TYPE ZFP_ARRAY_TYPE
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer1.cpp"
#include "template/cfparray.cpp"
#include "template/cfparray1.cpp"
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_view1f
#define ZFP_CONTAINER_TYPE zfp::array1f::view
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer1.cpp"
#include "template/cfpview.cpp"
#include "template/cfpview1.cpp"
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
