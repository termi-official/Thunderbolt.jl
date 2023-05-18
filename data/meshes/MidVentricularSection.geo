// Gmsh project created on Fri May  6 15:57:16 2022
SetFactory("OpenCASCADE");
//+
Cylinder(1) = {0, 0, 0, 0.4, 0, 0, 1.0, 2*Pi};
Cylinder(2) = {0, 0, 0, 0.4, 0, 0, 0.7, 2*Pi};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
//+
Rotate {{0, 1, 0}, {0, 0, 0}, Pi/2} {
  Volume{3};
}

//+
Physical Surface("Epicardium", 7) = {1};
//+
Physical Surface("Endocardium", 8) = {4};
//+
Physical Surface("Base", 9) = {2};
//+
Physical Surface("Myocardium", 10) = {3};
//+
Physical Volume("Myocardium", 11) = {3};
