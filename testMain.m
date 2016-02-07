%Main function to test videos based on existing model

wekaLoc = 'C:\Program Files\Weka-3-7\weka.jar';

%% location and paths to various files
[path,name,ext] = fileparts(mfilename('fullpath'));

%location of testing videos
testFileLoc = sprintf('%s\\testVideos\\',path);

%location where uniqY.mat is saved
uniqYloc = sprintf('%s\\resultFiles\\uniqY.mat',path);
%loaction of saved oriented rectangles
imLoc = sprintf('%s\\convolution\\',path)';
%path of arff file template 
templateLoc = sprintf('%s\\resultFiles\\template.arff',path);
%path of saved trained weka model
svModelLoc = sprintf('%s\\resultFiles\\model.mat',path);
%path to save predicted data in arff file format
testDbLoc = sprintf('%s\\resultFiles\\predict.arff',path);
%path where the predicted result of test viedos ares stored
testResLoc = sprintf('%s\\resultFiles\\testResult.txt',path);

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
%minimum frame number after which predictions take place
minFrameNo = 13;

%% function calls

test(templateLoc,testDbLoc,svModelLoc,imLoc,wekaLoc,testFileLoc,uniqYloc,testResLoc,height,width,alpha,boxsz,cropDivN,minFrameNo);