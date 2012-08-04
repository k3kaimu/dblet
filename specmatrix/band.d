/** 帯行列(バンドマトリックス)を操作するためのモジュールです
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */
module dblet.specmatrix.band;

struct BandMatrix(E){
public:
    size_t _rows;
    size_t _cols;
    size_t _kl;
    size_t _ku;
    E[] array;
    
    alias E elementType;
    
    ///コンストラクタ
    this(size_t r, size_t c, size_t kls, size_t kus){
        _rows = r;
        _cols = c;
        _kl = kls;
        _ku = kus;
        array = new E[n * (kl+ku+1)];
    }
    
    ///ditto
    this(size_t r, size_t c, size_t kls, size_t kus, E[] arr)
    in{
        assert(arr.length == (n * (kl+ku+1)));
    }
    body{
        _rows = r;
        _cols = c;
        _kl = kls;
        _ku = kus;
        array = arr;
    }
    
    
    E opIndex(size_t r, size_t c)
    in{
        assert(_rows > r);
        assert(_cols > c);
    }
    body{
        if(max(_cols, c - ku) <= r && r <=  min(_cols, c + kl))
            return array[j * (kl+ku+1) + ku+1+i-j];
        else
            return cast(E)0;
    }
    
    
    E opIndexAssign(E src, size_t r, size_t c)
    in{
        assert(_rows > r);
        assert(_cols > c);
        assert(max(_cols, c - ku) <= r && r <=  min(_cols, c+kl));
    }
    body{
        return array[j * (kl+ku+1) + ku+1+i-j] = src;
    }
    
    
    auto opBinary(string s : "*", S)(S src)
    if(is(S == E))
    {
        return bandMatrixExpr!("scal", E)(_rows, _cols, _kl, _ku, src, this);
    }
    
    auto opBinary(string s : "*", V)(V src)
    if(isVector!V && is(V.elementType == E))
    in{
        assert(_cols == src.dim);
    }
    body{
        static if(isVectorExpr!V && V.operator == "scal")
            return vectorExpr!("gbmv", E)(_rows, src.args[0], this, src.args[1]);
        else
            return vectorExpr!("gbmv", E)(_rows, 1, this, src);
    }
    
    auto opCast(M : Matrix!E)(){
        return matrixExpr!("band2normal", E)(_rows, _cols, this);
    }
    
}

unittest{
    BandMatrix!double(3,3,1,1);
}