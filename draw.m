
%% 1. load data
input50_ori47 = load('./MAT/input50_ori47.mat');
Cr = input50_ori47.Cr;

% global triangles_G;
tri_struct = load('./triangles.mat');
triangles = tri_struct.triangles;

% draw mesh image
faces_load = triangles; % vertices sequential number
faces_load = reshape(faces_load(:, 3), 4, [])'; % sparse operation get the four corner coordinates of the original mesh

%% 2. load trained data in python
f = 141.98903264492608;
R =  [[ 0.95104038  0.03898929 -0.3065975 ]
      [-0.05112743  0.99819032 -0.03165557]
      [ 0.30480843  0.04578127  0.95131272]];
  
t3d = [131.91925351 129.48883994   0.        ]';
alpha =[[ 7.23451203e-02]
 [-1.41828953e-01]
 [ 3.69445750e-01]
 [ 1.94977406e-01]
 [ 6.27724766e-02]
 [ 1.37964832e-01]
 [-1.08100473e-01]
 [-9.34464887e-02]
 [-2.69602075e-01]
 [ 1.22950893e-01]
 [ 8.46567924e-02]
 [ 4.63520606e-02]
 [-5.73879605e-02]
 [ 3.05146933e-02]
 [-1.68435807e-02]
 [-4.16251358e-02]
 [ 1.33106634e-04]
 [ 5.14391001e-02]
 [ 6.20402061e-02]
 [ 1.71699669e-02]
 [ 2.78477728e-02]
 [ 8.07909193e-03]
 [-2.74096629e-02]
 [-9.62466970e-03]
 [ 2.66755349e-02]
 [-7.29812099e-03]
 [-2.17541384e-03]
 [-4.05124014e-03]
 [ 1.40350560e-03]
 [-1.31640730e-02]
 [-7.04634149e-03]
 [-2.48463951e-04]
 [-1.21118612e-02]
 [-4.79542529e-03]
 [ 4.78209064e-03]
 [-1.67494110e-02]
 [-4.06310033e-03]
 [-4.17332213e-03]
 [ 9.28050948e-03]
 [-1.48011433e-02]
 [ 3.32365633e-03]
 [ 5.65096978e-03]
 [ 8.85300184e-03]
 [ 2.43826660e-03]
 [-3.27528724e-03]
 [-8.40295764e-04]
 [ 2.90880197e-03]
 [-6.05009252e-03]
 [ 6.82006816e-03]
 [-3.04658226e-03]];

alpha_exp = zeros(47,1);
alpha_exp(1) = 1;

%% 3. file reader and video writer
root_path = './probes/';
fileFolder = fullfile(root_path);
dirOutput = dir(fullfile(fileFolder, '*.jpg'));
dirPoints = dir(fullfile(fileFolder, '*lds87.txt'));
fileNames = {dirOutput.name}';
PointsNames = {dirPoints.name}';
one_img_path = char(strcat(root_path, fileNames(1)));
img = imread(one_img_path);
[height, width, nChannels] = size(img);
 
aviobj = VideoWriter('example_v6.avi');
aviobj.FrameRate = 30;
open(aviobj)
figure;
% gcf： the handle of the current figure
set(gcf, 'position', [0 0 width height]);

%% 4. transform
predict_all = ttm(Cr,{alpha',alpha_exp'},[2,3]); % dims, 34530 * 1; alpha, 50 * 1; alpha_exp, 47 * 1; cost 140 ms
pt3d_predict = reshape(predict_all.data,3,[]); % dims, 3 * 11510
index = find(pt3d_predict(3,:)>0); % dims, 1 * 5699
 
pt3d_predict = f * R * pt3d_predict + repmat(t3d, 1, size(pt3d_predict,2));
vertices_load = pt3d_predict';

[vertices_trim, faces_trim, move_list] = trimFace(vertices_load, faces_load, 1);
 
 %% 3d --> 2d
color_silver = [172/256 178/256 201/256]; % [207/256 216/256 243/256]
tcolor_trim_z =  repmat(color_silver, size(faces_trim, 1), 1);
vertices_load(:, 2) = height + 1 - vertices_load(:, 2); 
    
imshow(img, 'InitialMagnification', 'fit'), patch('vertices', vertices_load, 'faces', faces_trim, 'facevertexcdata', tcolor_trim_z, ...
                                                        'facecolor', 'flat', 'FaceAlpha', 1, 'EdgeAlpha', 1, 'EdgeColor', [0.5 0.5 0.5], ...
                                                        'Linewidth', 0.2, 'Linestyle', 'none', 'FaceLighting', 'gouraud', 'AmbientStrength', 0.80); 
material dull; % shiny, dull, metal; 
camlight headlight;  % headlight, right, left; 

writeVideo(aviobj,getframe(gcf));
   
close(aviobj)



