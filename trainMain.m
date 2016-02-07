%Main function to train a model based on input videos

wekaLoc = 'C:\Program Files\Weka-3-7\weka.jar';

%% location and paths to various files

[path,name,ext] = fileparts(mfilename('fullpath'));

%location of training videos
trFileLoc = sprintf('%s\\trainVideos\\',path);

%loaction of saved oriented rectangles
imLoc = sprintf('%s\\convolution\\',path)';
%location to save X.mat and Y.mat
svLoc = sprintf('%s\\resultFiles\\',path);
%location where Y.mat is saved
Yloc = sprintf('%s\\resultFiles\\Y.mat',path);
%location where uniqY.mat is saved
uniqYloc = sprintf('%s\\resultFiles\\uniqY.mat',path);
%location where X.mat is saved
Xloc = sprintf('%s\\resultFiles\\X.mat',path);
%path of arff file template 
templateLoc = sprintf('%s\\resultFiles\\template.arff',path);
%path to save trained data in arff file format
svArffLoc = sprintf('%s\\resultFiles\\train.arff',path);
dataLoc = svArffLoc;
%path to save trained weka model
svModelLoc = sprintf('%s\\resultFiles\\model.mat',path);

%% parameters

%height of oriented rectangles
height = 15;
%width of oriented rectangles
width = 5;
%orientation change
alpha = 15;
%maximum(height,width), where height and width are the dimensions to resize
%to bounding box
boxsz = 200;
%if N is cropDivN, then N*N is the number of divisions in the cropped
%bounding box
cropDivN = 3;
%minimum frame number after which training take place
minFrameNo = 13;

%% function calls

storeOrRect(height,width,alpha,imLoc);
makeDb(trFileLoc,imLoc,svLoc,height,width,alpha,boxsz,cropDivN,minFrameNo);
matToArff(Xloc,Yloc,uniqYloc,svArffLoc,templateLoc);
trModel(dataLoc,svModelLoc,wekaLoc);
