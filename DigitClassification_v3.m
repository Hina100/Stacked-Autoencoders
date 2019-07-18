clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();
clc
%%
% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
% rng('default')


% Load the training data into memory
%[xTrainImages, tTrain] = digittrain_dataset;
load('digittrain_dataset');
%% Parameters
epoch=100;
h1_size=150;
h2_size=75;
h3_size=25;

%% Layer 1
hiddenSize1 = h1_size;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',epoch, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% figure;
% plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrainImages);

%% Layer 2
hiddenSize2 = h2_size;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',epoch, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
%epoch =100
feat2 = encode(autoenc2,feat1);

%% Layer 3
hiddenSize3 = h3_size;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',epoch, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
%epoch =100
feat3 = encode(autoenc3,feat2);

%% Layer 4
% softnet = trainSoftmaxLayer(feat1,tTrain,'MaxEpochs',epoch)
% softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',epoch);
softnet = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',epoch);

%% Deep Net
%  deepnet = stack(autoenc1,softnet);
% deepnet = stack(autoenc1,autoenc2,softnet);
deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);
view(deepnet)

%% Test deep net
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
%[xTestImages, tTest] = digittest_dataset;
load('digittest_dataset');
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
y = deepnet(xTest);
% figure;
% plotconfusion(tTest,y);
classAcc_init=100*(1-confusion(tTest,y))


%% Test fine-tuned deep net (BACK PROPOGATION)
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
deepnet = train(deepnet,xTrain,tTrain);
y = deepnet(xTest);
% figure;
% plotconfusion(tTest,y);
classAcc_final=100*(1-confusion(tTest,y))
% view(deepnet)

%% plotting accuracy
% plot(MaxEpochs, classAcc_init, MaxEpochs, classAcc_final)
% xlabel('Number of epochs')
% ylabel('Accuracy')
% ylim([0,100])
% legend('initial accuracy', 'final accuracy')

%%
% Compare with normal neural network (1 hidden layers)
% net = patternnet(100);
% net=train(net,xTrain,tTrain);
% y=net(xTest);
% % plotconfusion(tTest,y);
% classAcc=100*(1-confusion(tTest,y))
% % view(net)
% 
% % Compare with normal neural network (2 hidden layers)
% net = patternnet([100 100]);
% net=train(net,xTrain,tTrain);
% y=net(xTest);
% % plotconfusion(tTest,y);
% classAcc=100*(1-confusion(tTest,y))
% % view(net)

%% added
% Compare with normal neural network (3 hidden layers)
% net = patternnet([100 100 100]);
% net=train(net,xTrain,tTrain);
% y=net(xTest);
% % plotconfusion(tTest,y);
% classAcc=100*(1-confusion(tTest,y))
% % view(net)
