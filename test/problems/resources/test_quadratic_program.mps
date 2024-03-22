* ENCODING=ISO-8859-1
NAME my problem
ROWS
 N  obj1    
 E  lin_eq  
 L  lin_leq 
 G  lin_geq 
 E  quad_eq 
 L  quad_leq
 G  quad_geq
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    x         obj1                            1
    x         lin_eq                          1
    x         lin_leq                         1
    x         lin_geq                         1
    x         quad_eq                         1
    x         quad_leq                        1
    x         quad_geq                        1
    y         obj1                           -1
    y         lin_eq                          2
    y         lin_leq                         2
    y         lin_geq                         2
    y         quad_eq                         1
    y         quad_leq                        1
    y         quad_geq                        1
    MARK0001  'MARKER'                 'INTEND'
    z         obj1                           10
RHS
    rhs       obj1                           -1
    rhs       lin_eq                          1
    rhs       lin_leq                         1
    rhs       lin_geq                         1
    rhs       quad_eq                         1
    rhs       quad_leq                        1
    rhs       quad_geq                        1
BOUNDS
 BV bnd       x       
 LO bnd       y                              -1
 UP bnd       y                               5
 LO bnd       z                              -1
 UP bnd       z                               5
QMATRIX
    x         x                               1
    y         z                              -1
    z         y                              -1
QCMATRIX   quad_eq
    x         x                               1
    y         z                            -0.5
    z         y                            -0.5
    z         z                               2
QCMATRIX   quad_leq
    x         x                               1
    y         z                            -0.5
    z         y                            -0.5
    z         z                               2
QCMATRIX   quad_geq
    x         x                               1
    y         z                            -0.5
    z         y                            -0.5
    z         z                               2
ENDATA
