#include "cfparray2f.h"
#include "zfparray2.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2f
#define CFP_REF_TYPE cfp_ref2f
#define CFP_PTR_TYPE cfp_ptr2f
#define CFP_ITER_TYPE cfp_iter2f
#define ZFP_ARRAY_TYPE zfp::array2f
#define ZFP_SCALAR_TYPE float

#define CFP_CONTAINER_TYPE CFP_ARRAY_TYPE
#define ZFP_CONTAINER_TYPE ZFP_ARRAY_TYPE
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer2.cpp"
#include "template/cfparray.cpp"
#include "template/cfparray2.cpp"
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_view2f
#define ZFP_CONTAINER_TYPE zfp::array2f::view
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer2.cpp"
#include "template/cfpview.cpp"
#include "template/cfpview2.cpp"
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
