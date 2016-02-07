function trModel(dataLoc,svModelLoc,wekaLoc)
%Function is used to train a weka model based on training data
%sataLoc - path of saved trained data in arff file format
%svModeloc - path to save trained weka model
%wekaLoc - path of weka files

%adding java files to the path of matlab
javaaddpath(wekaLoc);

%importing required weka files
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.core.*;
import java.io.*;

%creating trainModel as an object of weka.classifiers.trees.RandomForest()
trainModel = weka.classifiers.trees.RandomForest();

%loading dataset into ft_train_weka
loader = weka.core.converters.ArffLoader();
loader.setFile( java.io.File(dataLoc) );
ft_train_weka = loader.getDataSet();

%assigning last attribute as the class attribute
ft_train_weka.setClassIndex(ft_train_weka.numAttributes() - 1);

%building the classifier and getting a trained model
trainModel.setMaxDepth(0);
trainModel.setNumFeatures(0);
trainModel.setNumTrees(100);
trainModel.setSeed(1);
trainModel.buildClassifier(ft_train_weka);

%saving the trained model
save(svModelLoc,'trainModel');

%clearing all java objects
clear loader;
clear trainModel;
clear ft_train_weka;


