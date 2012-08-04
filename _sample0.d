import dblet.matrix;
import std.stdio;

pragma(lib, "blas");
pragma(lib, "dblet");
pragma(lib, "scid");

void main(){
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
     * alpha * A * B + beta * C
     * scal(alpha, A) * B + scal(beta, C)
     * gemm(alpha, A, B, 0, Dummy) + scal(beta, C)
     * gemm(alpha, A, B, beta, C)
     */
    
    writeln(typeof(ans).stringof);  //MatrixExpr!("gemm",double,double,Matrix!(double),Matrix!(double),double,Matrix!(double))
    writeln(ans.evalExpr);          //Matrix!(double)(3, 3, [21, -64, 210, 336, 514, 31, 70.8, 95, 47.5])
    //格納形式はcolumnメジャーなので、この順番で解は表示される。
}