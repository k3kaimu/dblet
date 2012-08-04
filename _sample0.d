import dblet.matrix;
import std.stdio;

pragma(lib, "blas");
pragma(lib, "dblet");
pragma(lib, "scid");    //現在はBLAS, LAPACKのbinding libraryとしてscidを使っているが、現在dblet.bindingに独自のbindingライブラリを作成中

void main(){

    //行列積
    auto A = Matrix!double(3, 3),
         B = Matrix!double(3, 3),
         C = Matrix!double(3, 3);
    
    //実際の格納形式はcolumnメジャーであるが、表向きの操作は数学で扱うようにrowメジャーな指定方法
    A[0,0] =1; A[0,1] = 8; A[0,2] = 3;
    A[1,0] =2; A[1,1] =10; A[1,2] = 8;
    A[2,0] =9; A[2,1] =-5; A[2,2] =-1;
    
    B[0,0] = 9; B[0,1] = 8; B[0,2] =3;
    B[1,0] = 3; B[1,1] =11; B[1,2] =2.3;
    B[2,0] =-8; B[2,1] = 6; B[2,2]=1;
    
    C[0,0] = 3; C[0,1] = 3; C[0,2] = 1.2;
    C[1,0] = 8; C[1,1] = 4; C[1,2] = 8;
    C[2,0] = 6; C[2,1] = 1; C[2,2] = -2;
    
    double alpha = 3, beta = -2;
    
    //演算は遅延評価的に行われ、この場合はコンパイル時簡約化によりblas level 3のgemmが実行される。
    auto ans = alpha * A * B + beta * C;
    /* 今回の場合のコンパイル時簡約化の手順
     * 1. alpha * A * B + beta * C
     * 2. scal(alpha, A) * B + scal(beta, C)
     * 3. gemm(alpha, A, B, 0, Dummy) + scal(beta, C)
     * 4. gemm(alpha, A, B, beta, C)
     */
    
    //ansの型を表示してみる
    writeln(typeof(ans).stringof);  //MatrixExpr!("gemm",double,double,Matrix!(double),Matrix!(double),double,Matrix!(double))
    
    //ansをそのまま表示しても遅延評価のため、演算結果は表示されない
    writeln(ans);   //MatrixExpr!("gemm",double,double,Matrix!(double),Matrix!(double),double,Matrix!(double))(3, 3, 3, Matrix!(double)(3, 3, [1, 2, 9, 8, 10, -5, 3, 8, -1]), Matrix!(double)(3, 3, [9, 3, -8, 8, 11, 6, 3, 2.3, 1]), -2, Matrix!(double)(3, 3, [3, 8, 6, 3, 4, 1, 1.2, 8, -2]))
    
    //ansを実際に計算してみる
    writeln(ans.evalExpr);          //Matrix!(double)(3, 3, [21, -64, 210, 336, 514, 31, 70.8, 95, 47.5])
    //解の格納形式ももちろんcolumnメジャー
    
    writeln(ans[0, 0]);             //対角要素だけ計算してみる
    writeln(ans[1, 1]);             //遅延評価されているので、他の要素は演算されない
    writeln(ans[2, 2]);             //そのため、全体が必要なければ(部分だけでいいのであれば)高速
}