/** BLAS binding library for D.
 * Authors: Komatsu Kazuki (k3kaimu)
 * Date: July 29, 2012
 * License: NYSL
 */
module dblet.binding.blas;

import std.typetuple    : staticIndexOf;

///BLASの型.任意の型について設定します。
template Types(S, D, C, Z, Ch, I)
{
    alias S Single;
    alias D Double;
    alias C SCmplx;
    alias Z DCmplx;
    alias Ch Char;
    alias I Int;
}

///ditto
alias Types!(float, double, cfloat, cdouble, char, int) Blas;

///ditto
alias TypeTuple!(Blas.Single, Blas.Double,
                 Blas.SCmplx, Blas.DCmplx) BlasTypes;

///Tに対応する実数型
template RealType(T){
    static if(is(T == Blas.SCmplx))
        alias Blas.Single RealType;
    else static if(is(T == Blas.DCmplx))
        alias Blas.Double RealType;
    else
        alias T RealType;
}

//関数についてパターンマッチを行う
private template PatternMatch(fun...){
    static if(fun.length){
        auto PatternMatch(T...)(T args){
            static if(is(typeof(fun[0](args))))
                return fun[0](args);
            else
                return .PatternMatch!(fun[1..$])(args);
        }
    }else{
        static assert(0, "Cannot match for patterns");
    }
}

extern(C):

//level 1

/** 以下の式のように、要素数nのベクトルxに対してその成分の大きさ(複素ベクトルの場合は実部・虚部の大きさの和)の合計を返します。
 * result = Σ(i = 0 -> n-1){ |Re{x[i*incx]}| + |Im{x[i*incx]}| }
 * 
 * Returns:     上式のresult
 * Params:
 *      n =     xベクトルの要素数へのポインタ
 *      x =     xベクトルの先頭要素へのポインタ
 *      incx =  xベクトルのインクリメント幅へのポインタ。普通は1へのポインタ
 */
Blas.Single  sasum_(Blas.Int* n, Blas.Single* x, Blas.Int* incx);
Blas.Double  dasum_(Blas.Int* n, Blas.Double* x, Blas.Int* incx);   ///ditto
Blas.Single scasum_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx);   ///ditto
Blas.Double dzasum_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx);   ///ditto

///ditto
alias PatternMatch!(sasum_, dasum_, scasum_, dzasum_) asum_;

///ditto
auto asum(T)(Blas.Int n, T[] x, Blas.Int incx)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    return asum_(&n, x.ptr, &incx);
}


/** 以下の式のように、ベクトルxにスカラーaをかけ、ベクトルyに加算します。
 * y := a * x + y
 * 
 * Returns:     上式の結果のyベクトルがyに格納されます。
 * Params:
 *      n =     ベクトルx, yの次数へのポインタ
 *      a =     スカラー値
 *      x =     ベクトルxの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incx|)
 *      incx =  xのインクリメント幅へのポインタ。通常は1へのポインタ
 *      y =     ベクトルyの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incy|)
 *      incy =  yのインクリメント幅へのポインタ。通常は1へのポインタ
 */
void saxpy_(Blas.Int* n, Blas.Single* a, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy);
void daxpy_(Blas.Int* n, Blas.Double* a, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy);   ///ditto
void caxpy_(Blas.Int* n, Blas.SCmplx* a, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy);   ///ditto
void zaxpy_(Blas.Int* n, Blas.DCmplx* a, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy);   ///ditto

///ditto
alias PatternMatch!(saxpy_, daxpy_, caxpy_, zaxpy_) axpy_;

///ditto
void axpy(T)(Blas.Int n, T[] a, T[] x, Blas.Int incx, T[] y, Blas.Int incy)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    axpy_(&n, a.ptr, x.ptr, &incx, y.ptr, incy);
}


/** ベクトルxをベクトルyにコピーします。
 * y = x
 * 
 * Returns:     xのコピーがyに格納されます。
 * Params:
 *      n =     ベクトルx, yの次数へのポインタ
 *      x =     ベクトルxの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incx|)
 *      incx =  xのインクリメント幅へのポインタ。通常は1へのポインタ
 *      y =     ベクトルyの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incy|)
 *      incy =  yのインクリメント幅へのポインタ。通常は1へのポインタ    
 */
void scopy_(Blas.Int* n, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy);
void dcopy_(Blas.Int* n, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy);   ///ditto
void ccopy_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy);   ///ditto
void zcopy_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy);   ///ditto

///ditto
alias PatternMatch!(scopy_, dcopy_, ccopy_, zcopy_) copy_;

///ditto
void copy(T)(Blas.Int n, T[] x, Blas.Int incx, T[] y, Blas.Int incy)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    copy_(&n, x.ptr, &incx, y.ptr, &incy);
}


/** 2ベクトルx, yのドット積を計算します。
 * result = Σ(x*y)
 * 
 * Returns:     xとyのドット積(上式のresult)
 * Params:
 *      n =     ベクトルx, yの次数へのポインタ
 *      x =     ベクトルxの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incx|)
 *      incx =  xのインクリメント幅へのポインタ。通常は1へのポインタ
 *      y =     ベクトルyの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incy|)
 *      incy =  yのインクリメント幅へのポインタ。通常は1へのポインタ    
 */
Blas.Single  sdot_(Blas.Int* n, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy);
Blas.Double  ddot_(Blas.Int* n, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy);    ///ditto
Blas.SCmplx cdotu_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy);    ///ditto
Blas.DCmplx zdotu_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy);    ///ditto

///ditto
alias PatternMatch!(sdot_, ddot_, cdotu_, zdotu_) dot_;

///ditto
T dot(T)(Blas.Int n, T[] x, Blas.Int incx, T[] y, Blas.Int incy)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    return dot_(&n, x.ptr, &incx, y.ptr, &incy);
}


/** ベクトルxの共役とベクトルyのドット積を計算します。
 * result = Σ(conjg(x)*y)
 * 
 * Returns:     xとyのドット積(上式のresult)
 * Params:
 *      n =     ベクトルx, yの次数へのポインタ
 *      x =     ベクトルxの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incx|)
 *      incx =  xのインクリメント幅へのポインタ。通常は1へのポインタ
 *      y =     ベクトルyの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incy|)
 *      incy =  yのインクリメント幅へのポインタ。通常は1へのポインタ    
 */
Blas.SCmplx cdotc_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy);
Blas.DCmplx zdotc_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy);    ///ditto

///ditto
alias PatternMatch!(cdotc_, zdotc_) dotc_;

///ditto
T dot(T)(Blas.Int n, T[] x, Blas.Int incx, T[] y, Blas.Int incy)
if(staticIndexOf!(T, BlasTypes[2..$]) != -1)
{
    return dotc_(&n, x.ptr, &incx, y.ptr, &incy);
}


/** ベクトルxのユークリッド・ノルムを計算します。
 * result = ||x||2 = √( Σ( |x[i]| ^^ 2) )
 * 
 * Returns:     ベクトルxのユークリッド・ノルム
 * Params:
 *      n =     ベクトルxの次数へのポインタ
 *      x =     ベクトルxの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incx|)
 *      incx =  xのインクリメント幅へのポインタ。通常は1へのポインタ 
 */
Blas.Single  snrm2_(Blas.Int* n, Blas.Single* x, Blas.Int* incx);
Blas.Double  dnrm2_(Blas.Int* n, Blas.Double* x, Blas.Int* incx);   ///ditto
Blas.Single scnrm2_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx);   ///ditto
Blas.Double dznrm2_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx);   ///ditto

///ditto
alias PatternMatch!(snrm2_, dnrm2_, scnrm2_, dznrm2_) nrm2_;

///ditto
auto nrm2(T)(Blas.Int n, T[] x, Blas.Int incx)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    return nrm2_(&n, x.ptr, &incx);
}


/** ベクトルxとベクトルyで、点pを p[i] = (x[i], y[i]) と表し、c = cosθ, s = sinθとなるθだけ、各p点を回転させます。式で表すと以下のようになります。
 * x[i] := c * x[i] + s * y[i]
 * y[i] :=-s * x[i] + c * y[i]
 * 
 * Returns:     上式の結果がxとyに格納される。
 * Params:
 *      n =     ベクトルx, yの次数へのポインタ
 *      x =     ベクトルxの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incx|)
 *      incx =  xのインクリメント幅へのポインタ。通常は1へのポインタ
 *      y =     ベクトルyの先頭要素へのポインタ。大きさは少なくとも(1 + (n-1) * |incy|)
 *      incy =  yのインクリメント幅へのポインタ。通常は1へのポインタ  
 *      c =     実数。通常はconθの値
 *      s =     実数。通常はsinθの値
 */
void  srot_(Blas.Int* n, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy, Blas.Single* c, Blas.Single* s);
void  drot_(Blas.Int* n, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy, Blas.Double* c, Blas.Double* s);   ///ditto
void csrot_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy, Blas.Single* c, Blas.Single* s);   ///ditto
void zdrot_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy, Blas.Double* c, Blas.Double* s);   ///ditto

///ditto
alias PatternMatch!(srot_, drot_, csrot_, zdrot_) rot_;

///ditto
void rot(T, U : RealType!T)(Blas.Int n, T[] x, Blas.Int incx, T[] y, Blas.Int incy, U c, U s)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    rot_(&n, x.ptr, &incx, y.ptr, &incy, &c, &s);
}


/** 点p(a, b)を与えたとき、この点をGivens回転させた結果のy座標が0になるような各パラメータa, b, c, sを計算します。
 * r = ||(a, b)||
 * c * a + s * b = r
 * -s* a + c * b = 0
 * c^^2 + s^^2 = 1
 * z = | if(|a| >  |b|)                       = s
 *     | if(|a| <= |b| and c != 0 and r != 0) = 1/c 
 * となるようなr, z, c, sを求めます。 
 *
 * Returns:     aに求めたrが、bに求めたzが、cに求めたcが、sに求めたsが格納されます。
 * Params:
 *      a =     点pのx座標。求めたrが格納される
 *      b =     点pのy座標。求めたzが格納される
 *      c =     求めたcが格納される
 *      s =     求めたsが格納される
 */
void srotg_(Blas.Single* a, Blas.Single* b, Blas.Single* c, Blas.Single* s);
void drotg_(Blas.Double* a, Blas.Double* b, Blas.Double* c, Blas.Double* s);    ///ditto
void crotg_(Blas.SCmplx* a, Blas.SCmplx* b, Blas.Single* c, Blas.Single* s);    ///ditto
void zrotg_(Blas.DCmplx* a, Blas.DCmplx* b, Blas.Double* c, Blas.Double* s);    ///ditto

///ditto
alias PatternMatch!(srotg_, drotg_, crotg_, zrotg_) rotg_;

///ditto
void rotg(T, U : RealType!T)(ref T a, ref T b, ref U c, ref U s)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    rotg_(&a, &b, &s, &c);
}


///変形面における点の回転を実行する。
void srotm_(Blas.Int* n, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy, Blas.Single* param);
void drotm_(Blas.Int* n, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy, Blas.Double* param);   ///ditto

///ditto
alias PatternMatch!(srotm_, drotm_) rotm_;

///ditto
void rotm(T)(Blas.Int n, T[] x, Blas.Int incx, T[] y, Blas.Int incy, ref Blas.Single[5] param)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    rotm_(&n, x.ptr, &incx, y.ptr, &incy, param.ptr);
}


/// Givens回転に対する変形パラメータを計算する。
void srotmg_(Blas.Single* d1, Blas.Single* d2, Blas.Single* b1, Blas.Single* b2, Blas.Single* param);
void drotmg_(Blas.Double* d1, Blas.Double* d2, Blas.Double* b1, Blas.Double* b2, Blas.Double* param);   ///ditto


///ditto
alias PatternMatch!(srotmg_, drotmg_) rotmg_;

///ditto
void rotmg(T)(T d1, T d2, T b1, T b2, ref T[5] param)
{
    rotmg_(&d1, &d2, &b1, &b2, param.ptr);
}


///ベクトルとスカラーの積を計算する。
void  sscal_(Blas.Int* n, Blas.Single* alpha, Blas.Single* x, Blas.Int* incx);
void  dscal_(Blas.Int* n, Blas.Double* alpha, Blas.Double* x, Blas.Int* incx);  ///ditto
void  cscal_(Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* x, Blas.Int* incx);  ///ditto
void csscal_(Blas.Int* n, Blas.Single* alpha, Blas.SCmplx* x, Blas.Int* incx);  ///ditto
void  zscal_(Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* x, Blas.Int* incx);  ///ditto
void zdscal_(Blas.Int* n, Blas.Double* alpha, Blas.DCmplx* x, Blas.Int* incx);  ///ditto

///ditto
alias PatternMatch!(sscal_, dscal_, cscal_, csscal_, zscal_, zdscal_) scal_;

///ditto
void scal(T, U)(Blas.Int n, T alpha, U[] x, Blas.Int incx)
if(staticIndexOf!(T, BlasTypes) != -1 && staticIndexOf!(U, BlasTypes))
{
    scal_(&n, &alpha, x.ptr, &incx);
}


///ベクトルを他のベクトルと交換する。
void sswap_(Blas.Int* n, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy);
void dswap_(Blas.Int* n, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy);   ///ditto
void cswap_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy);   ///ditto
void zswap_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy);   ///ditto

///ditto
alias PatternMatch!(sswap_, dswap_, cswap_, zswap_) swap_;

///ditto
void swap(T)(Blas.Int n, T[] x, Blas.Int incx, T[] y, Blas.Int incy)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    swap_(&n, x.ptr, &incx, y.ptr, &incy);
}


///最大絶対値を持つベクトル成分x[i]の位置を返す。複素ベクトルの場合は最大合計 |Re{x[i]}| + |Im{x[i]}| を持つ成分の位置を返す。
Blas.Int isamax_(Blas.Int* n, Blas.Single* x, Blas.Int* incx);
Blas.Int idamax_(Blas.Int* n, Blas.Double* x, Blas.Int* incx);  ///ditto
Blas.Int icamax_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx);  ///ditto
Blas.Int izamax_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx);  ///ditto

///ditto
alias PatternMatch!(isamax_, idamax_, icamax_, izamax_) iamax_;

///ditto
Blas.Int iamax(T)(Blas.Int n, T[] x, Blas.Int incx)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    return iamax_(&n, x.ptr, &incx);
}


///最小絶対値を持つベクトル成分x[i]の位置を返す。複素ベクトルの場合は最小合計 |Re{x[i]}| + |Im{x[i]}| を持つ成分の位置を返す。
Blas.Int isamax_(Blas.Int* n, Blas.Single* x, Blas.Int* incx);
Blas.Int idamax_(Blas.Int* n, Blas.Double* x, Blas.Int* incx);  ///ditto
Blas.Int icamax_(Blas.Int* n, Blas.SCmplx* x, Blas.Int* incx);  ///ditto
Blas.Int izamax_(Blas.Int* n, Blas.DCmplx* x, Blas.Int* incx);  ///ditto

///ditto
alias PatternMatch!(isamin_, idamin_, icamin_, izamin_) iamin_;

///ditto
Blas.Int iamin(T)(Blas.Int n, T[] x, Blas.Int incx)
if(staticIndexOf!(T, BlasTypes) != -1)
{
    return iamin_(&n, x.ptr, &incx);
}


//level 2
///一般バンド・行列での行列・ベクトル積を計算します。
void sgbmv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.Int* kl, Blas.Int* ku, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Single* beta, Blas.Single* y, Blas.Int* incy, Blas.Int trans_len);
void dgbmv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.Int* kl, Blas.Int* ku, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Double* beta, Blas.Double* y, Blas.Int* incy, Blas.Int trans_len);  ///ditto
void cgbmv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.Int* kl, Blas.Int* ku, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* beta, Blas.SCmplx* y, Blas.Int* incy, Blas.Int trans_len);  ///ditto
void zgbmv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.Int* kl, Blas.Int* ku, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* beta, Blas.DCmplx* y, Blas.Int* incy, Blas.Int trans_len);  ///ditto

///一般行列での行列・ベクトル積を計算します。
void sgemv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Single* beta, Blas.Single* y, Blas.Int* incy, Blas.Int trans_len);
void dgemv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Double* beta, Blas.Double* y, Blas.Int* incy, Blas.Int trans_len);  ///ditto
void cgemv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* beta, Blas.SCmplx* y, Blas.Int* incy, Blas.Int trans_len);  ///ditto
void zgemv_(Blas.Char* trans, Blas.Int* m, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* beta, Blas.DCmplx* y, Blas.Int* incy, Blas.Int trans_len);  ///ditto


///列ベクトルと行ベクトルの積を計算し、alpha倍の値をA行列に加算します。
void sger_(Blas.Int* m, Blas.Int* n, Blas.Single* alpha, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy, Blas.Single* A, Blas.Int* lda);
void dger_(Blas.Int* m, Blas.Int* n, Blas.Double* alpha, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy, Blas.Double* A, Blas.Int* lda);    ///ditto
void cgeru_(Blas.Int* m, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy, Blas.SCmplx* A, Blas.Int* lda);   ///ditto
void zgeru_(Blas.Int* m, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy, Blas.DCmplx* A, Blas.Int* lda);   ///ditto

///列ベクトルと行ベクトルの共役ベクトルの積を計算し、alpha倍の値をA行列に加算します。
void cgerc_(Blas.Int* m, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy, Blas.SCmplx* A, Blas.Int* lda);
void zgerc_(Blas.Int* m, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy, Blas.DCmplx* A, Blas.Int* lda);   ///ditto

///エルミート・バンド・行列を使用して、行列・ベクトルの積を計算します。
void chbmv_(Blas.Char* uplo, Blas.Int* n, Blas.Int* k, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* beta, Blas.SCmplx* y, Blas.Int* incy, Blas.Int uplo_len);
void zhbmv_(Blas.Char* uplo, Blas.Int* n, Blas.Int* k, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* beta, Blas.DCmplx* y, Blas.Int* incy, Blas.Int uplo_len);    ///ditto

///エルミート・行列を使用して、行列・ベクトルの積を計算します。
void chemv_(Blas.Char* uplo, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* beta, Blas.SCmplx* y, Blas.Int* incy, Blas.Int uplo_len);
void zhemv_(Blas.Char* uplo, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* beta, Blas.DCmplx* y, Blas.Int* incy, Blas.Int uplo_len); ///ditto

///列ベクトルとその共役転置ベクトルの積を計算します。
void cher_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* A, Blas.Int* lda, Blas.Int uplo_len); 
void zher_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* A, Blas.Int* lda, Blas.Int uplo_len); ///ditto

///エルミート・行列の階数2の更新を実行する。a := alpha*x*conjg(y') + conjg(alpha)*y*conjg(x')
void cher2_(Blas.Char* uplo, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy, Blas.SCmplx* A, Blas.Int* lda, Blas.Int uplo_len);
void zher2_(Blas.Char* uplo, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy, Blas.DCmplx* A, Blas.Int* lda, Blas.Int uplo_len);    ///ditto

///パックド形式のエルミート・行列を使用して行列・ベクトルの積を計算します。
void chpmv_(Blas.Char* uplo, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* beta, Blas.SCmplx* y, Blas.Int* incy, Blas.Int uplo_len);
void zhpmv_(Blas.Char* uplo, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* beta, Blas.DCmplx* y, Blas.Int* incy, Blas.Int uplo_len);    ///ditto

///パックド形式のエルミート・行列の階数1の更新を実行します。
void chpr_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* A, Blas.Int uplo_len);
void zhpr_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* A, Blas.Int uplo_len);    ///ditto

///パックド形式のエルミート・行列の階数2の更新を実行します。
void chpr2_(Blas.Char* uplo, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* x, Blas.Int* incx, Blas.SCmplx* y, Blas.Int* incy, Blas.SCmplx* A, Blas.Int uplo_len);
void zhpr2_(Blas.Char* uplo, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* x, Blas.Int* incx, Blas.DCmplx* y, Blas.Int* incy, Blas.DCmplx* A, Blas.Int uplo_len);   ///ditto

///対称バンド・行列を使用して、行列・ベクトルの積を計算します。
void ssbmv_(Blas.Char* uplo, Blas.Int* n, Blas.Int* k, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Single* beta, Blas.Single* y, Blas.Int* incy, Blas.Int uplo_len);
void dsbmv_(Blas.Char* uplo, Blas.Int* n, Blas.Int* k, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Double* beta, Blas.Double* y, Blas.Int* incy, Blas.Int uplo_len);    ///ditto

///パックド形式の対称行列を使用して行列・ベクトルの積を計算します。
void sspmv_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.Single* ap, Blas.Single* x, Blas.Int* incx, Blas.Single* beta, Blas.Single* y, Blas.Int* incy, Blas.Int uplo_len);
void dspmv_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.Double* ap, Blas.Double* x, Blas.Int* incx, Blas.Double* beta, Blas.Double* y, Blas.Int* incy, Blas.Int uplo_len);   ///ditto

///パックド形式の対称行列の階数1の更新を実行します。
void sspr_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.Single* x, Blas.Int* incx, Blas.Single* ap, Blas.Int uplo_len);
void dspr_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.Double* x, Blas.Int* incx, Blas.Double* ap, Blas.Int uplo_len);   ///ditto

///パックド形式の対称行列の階数2の更新を実行します。
void sspr2_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy, Blas.Single* ap, Blas.Int uplo_len);
void dspr2_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy, Blas.Double* ap, Blas.Int uplo_len);  ///ditto

///対称行列の階数1の更新を実行します。
void ssyr_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.Single* x, Blas.Int* incx, Blas.Single* A, Blas.Int* lda, Blas.Int uplo_len);
void dsyr_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.Double* x, Blas.Int* incx, Blas.Double* A, Blas.Int* lda, Blas.Int uplo_len); ///ditto

///対称行列に対して行列・ベクトルの積を計算します。
void ssymv_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Single* beta, Blas.Single* y, Blas.Int* incy, Blas.Int uplo_len);
void dsymv_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Double* beta, Blas.Double* y, Blas.Int* incy, Blas.Int uplo_len); ///ditto

///対称行列の階数2の更新を実行します。
void ssyr2_(Blas.Char* uplo, Blas.Int* n, Blas.Single* alpha, Blas.Single* x, Blas.Int* incx, Blas.Single* y, Blas.Int* incy, Blas.Single* A, Blas.Int* lda, Blas.Int uplo_len);
void dsyr2_(Blas.Char* uplo, Blas.Int* n, Blas.Double* alpha, Blas.Double* x, Blas.Int* incx, Blas.Double* y, Blas.Int* incy, Blas.Double* A, Blas.Int* lda, Blas.Int uplo_len);    ///ditto

///三角バンド・行列を使用して、行列・ベクトルの積を計算します。
void stbmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);
void dtbmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len); ///ditto
void ctbmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len); ///ditto
void ztbmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len); ///ditto

///係数が三角バンド行列に格納されている連立1次方程式を解く。
void stbsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);
void dtbsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len); ///ditto
void ctbsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len); ///ditto
void ztbsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Int* k, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len); ///ditto

///パックド形式の三角行列を使用して行列・ベクトルの積を計算します。
void stpmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Single* ap, Blas.Single* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);
void dtpmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Double* ap, Blas.Double* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);    ///ditto
void ctpmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.SCmplx* ap, Blas.SCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);    ///ditto
void ztpmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.DCmplx* ap, Blas.DCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);    ///ditto

///係数がパックド形式の三角行列に格納されている連立1次方程式を解きます。
void stpsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Single* ap, Blas.Single* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);
void dtpsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Double* ap, Blas.Double* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);    ///ditto
void ctpsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.SCmplx* ap, Blas.SCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);    ///ditto
void ztpsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.DCmplx* ap, Blas.DCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);    ///ditto

///三角行列を使用して、行列・ベクトル積を計算します。
void strmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);
void dtrmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);  ///ditto
void ctrmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);  ///ditto
void ztrmv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);  ///ditto

///係数が三角行列に格納されている連立1次方程式を解く。
void strsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Single* A, Blas.Int* lda, Blas.Single* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);
void dtrsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.Double* A, Blas.Int* lda, Blas.Double* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);  ///ditto
void ctrsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);  ///ditto
void ztrsv_(Blas.Char* uplo, Blas.Char* trans, Blas.Char* diag, Blas.Int* n, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* x, Blas.Int* incx, Blas.Int uplo_len, Blas.Int trans_len, Blas.Int diag_len);  ///ditto


//level 3

///スカラー・行列・行列の積を計算し、その結果をスカラー・行列の積に加える。
void sgemm_(Blas.Char* transa, Blas.Char* transb, Blas.Int* m, Blas.Int* n, Blas.Int* k, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* B, Blas.Int* ldb, Blas.Single* beta, Blas.Single* C, Blas.Int* ldc, Blas.Int transa_len, Blas.Int transb_len);
void dgemm_(Blas.Char* transa, Blas.Char* transb, Blas.Int* m, Blas.Int* n, Blas.Int* k, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* B, Blas.Int* ldb, Blas.Double* beta, Blas.Double* C, Blas.Int* ldc, Blas.Int transa_len, Blas.Int transb_len); ///ditto
void cgemm_(Blas.Char* transa, Blas.Char* transb, Blas.Int* m, Blas.Int* n, Blas.Int* k, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* B, Blas.Int* ldb, Blas.SCmplx* beta, Blas.SCmplx* C, Blas.Int* ldc, Blas.Int transa_len, Blas.Int transb_len); ///ditto
void zgemm_(Blas.Char* transa, Blas.Char* transb, Blas.Int* m, Blas.Int* n, Blas.Int* k, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* B, Blas.Int* ldb, Blas.DCmplx* beta, Blas.DCmplx* C, Blas.Int* ldc, Blas.Int transa_len, Blas.Int transb_len); ///ditto

///スカラー・行列・行列の積を計算し ( 行列のいずれかはエルミート行列 )、その結果をスカラー・行列の積に加える。
void chemm_(Blas.Char* side, Blas.Char* uplo, Blas.Int* m, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* B, Blas.Int* ldb, Blas.SCmplx* beta, Blas.SCmplx* C, Blas.Int* ldc, Blas.Int side_len, Blas.Int uplo_len);
void zhemm_(Blas.Char* side, Blas.Char* uplo, Blas.Int* m, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* B, Blas.Int* ldb, Blas.DCmplx* beta, Blas.DCmplx* C, Blas.Int* ldc, Blas.Int side_len, Blas.Int uplo_len);  ///ditto

///エルミート・行列の階数nの更新を実行する。
void cherk_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.Single* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.Single* beta, Blas.SCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);
void zherk_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.Double* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.Double* beta, Blas.DCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto

///エルミート・行列の階数2kの更新を実行する。
void cher2k_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* B, Blas.Int* ldb, Blas.Single* beta, Blas.SCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);
void zher2k_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* B, Blas.Int* ldb, Blas.Double* beta, Blas.DCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto


///スカラー・行列・行列の積 ( 行列のいずれかは対称行列 ) を計算し、その結果をスカラー・行列の積に加える。
void ssymm_(Blas.Char* side, Blas.Char* uplo, Blas.Int* m, Blas.Int* n, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* B, Blas.Int* ldb, Blas.Single* beta, Blas.Single* C, Blas.Int* ldc, Blas.Int side_len, Blas.Int uplo_len);
void dsymm_(Blas.Char* side, Blas.Char* uplo, Blas.Int* m, Blas.Int* n, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* B, Blas.Int* ldb, Blas.Double* beta, Blas.Double* C, Blas.Int* ldc, Blas.Int side_len, Blas.Int uplo_len);  ///ditto
void csymm_(Blas.Char* side, Blas.Char* uplo, Blas.Int* m, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* B, Blas.Int* ldb, Blas.SCmplx* beta, Blas.SCmplx* C, Blas.Int* ldc, Blas.Int side_len, Blas.Int uplo_len);  ///ditto
void zsymm_(Blas.Char* side, Blas.Char* uplo, Blas.Int* m, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* B, Blas.Int* ldb, Blas.DCmplx* beta, Blas.DCmplx* C, Blas.Int* ldc, Blas.Int side_len, Blas.Int uplo_len);  ///ditto


///対称行列の階数 n の更新を実行する。
void ssyrk_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* beta, Blas.Single* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);
void dsyrk_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* beta, Blas.Double* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto
void csyrk_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* beta, Blas.SCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto
void zsyrk_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* beta, Blas.DCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto


///対称行列の階数 2 の更新を実行する。
void ssyr2k_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* B, Blas.Int* ldb, Blas.Single* beta, Blas.Single* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);
void dsyr2k_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* B, Blas.Int* ldb, Blas.Double* beta, Blas.Double* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto
void csyr2k_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* B, Blas.Int* ldb, Blas.SCmplx* beta, Blas.SCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto
void zsyr2k_(Blas.Char* uplo, Blas.Char* trans, Blas.Int* n, Blas.Int* k, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* B, Blas.Int* ldb, Blas.DCmplx* beta, Blas.DCmplx* C, Blas.Int* ldc, Blas.Int uplo_len, Blas.Int trans_len);   ///ditto

///スカラー・行列・行列の積 ( 行列のいずれかは三角行列 ) を計算する。
void strmm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);
void dtrmm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);    ///ditto
void ctrmm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);    ///ditto
void ztrmm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);    ///ditto

///行列式 ( 行列のいずれかは三角行列 ) を解く。
void strsm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.Single* alpha, Blas.Single* A, Blas.Int* lda, Blas.Single* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);
void dtrsm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.Double* alpha, Blas.Double* A, Blas.Int* lda, Blas.Double* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);    ///ditto
void ctrsm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.SCmplx* alpha, Blas.SCmplx* A, Blas.Int* lda, Blas.SCmplx* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);    ///ditto
void ztrsm_(Blas.Char* side, Blas.Char* uplo, Blas.Char* transa, Blas.Char* diag, Blas.Int* m, Blas.Int* n, Blas.DCmplx* alpha, Blas.DCmplx* A, Blas.Int* lda, Blas.DCmplx* B, Blas.Int* ldb, Blas.Int side_len, Blas.Int uplo_len, Blas.Int transa_len, Blas.Int diag_len);    ///ditto

