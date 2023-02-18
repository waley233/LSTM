%% 1.环境清理
clear, clc, close all;
%% 2.导入数据
data=xlsread('1.csv');
data1=data;
% 原始数据绘图
figure
plot(data,'-s','Color',[0 0 255]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
legend('Original Data','Location','NorthWest','FontName','Times New Roman');
xlabel('Sample','fontsize',12,'FontName','Times New Roman');
ylabel('Value','fontsize',12,'FontName','Times New Roman');
%% 3.数据处理
numTimeStepsTrain = floor(350);%数据训练 ，剩下的用来验证
[XTrain,YTrain,XTest,YTest,mu,sig] = shujuchuli(data,numTimeStepsTrain);
%% 4.定义LSTM结构参数
numFeatures= 1;%输入节点
numResponses = 1;%输出节点
numHiddenUnits = 500;%隐含层神经元节点数 

%构建 LSTM网络 
layers = [sequenceInputLayer(numFeatures) 
 lstmLayer(numHiddenUnits) %lstm函数 
dropoutLayer(0.2)%丢弃层概率 
 reluLayer('name','relu')% 激励函数 RELU 
fullyConnectedLayer(numResponses)
regressionLayer];

XTrain=XTrain';
YTrain=YTrain';

%% 5.定义LSTM函数参数 
def_options();
%% 6.训练LSTM网络 
net = trainNetwork(XTrain,YTrain,layers,options);

%% 7.建立训练模型 
net = predictAndUpdateState(net,XTrain);

%% 8.仿真预测(训练集) 
M = numel(XTrain);
for i = 1:M
    [net,YPred_1(:,i)] = predictAndUpdateState(net,XTrain(:,i),'ExecutionEnvironment','cpu');%
end
T_sim1 = sig*YPred_1 + mu;%预测结果去标准化 ，恢复原来的数量级 
%% 9.仿真预测(验证集) 
N = numel(XTest);
for i = 1:N
    [net,YPred_2(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');%
end
T_sim2 = sig*YPred_2 + mu;%预测结果去标准化 ，恢复原来的数量级 
%% 10.评价指标
%  均方根误差
T_train=data1(1:M)';
T_test=data1(M+1:end)';
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['The MAE of training set：', num2str(mae1)])
disp(['The MAE of verification set：', num2str(mae2)])
%  MAPE
maep1 = sum(abs(T_sim1 - T_train)./T_train) ./ M ;
maep2 = sum(abs(T_sim2 - T_test )./T_test) ./ N ;
disp(['The MAPE of training set：', num2str(maep1)])
disp(['The MAPE of verification set：', num2str(maep2)])
%  RMSE
RMSE1 = sqrt(sumsqr(T_sim1 - T_train)/M);
RMSE2 = sqrt(sumsqr(T_sim2 - T_test)/N);
disp(['The RMSE of training set：', num2str(RMSE1)])
disp(['The RMSE of verification set：', num2str(RMSE2)])
%% 11. 绘图
figure
subplot(2,1,1)
plot(T_sim1,'-s','Color',[255 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[250 0 0]./255)
hold on 
plot(T_train,'-o','Color',[150 150 150]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[150 150 150]./255)
legend( 'LSTM fitting the training data','Actual analytical data','Location','NorthWest','FontName','Times New Roman');
title('The prediciting result of LSTM model and the true value','fontsize',12,'FontName','Times New Roman')
xlabel('Sample','fontsize',12,'FontName','Times New Roman');
ylabel('Data','fontsize',12,'FontName','Times New Roman');
xlim([1 M])
%-------------------------------------------------------------------------------------
subplot(2,1,2)
bar((T_sim1 - T_train)./T_train)   
legend('The relative error of LSTM model training set','Location','NorthEast','FontName','Times New Roman')
title('The relative error of LSTM model training set','fontsize',12,'FontName','Times New Roman')
ylabel('Error','fontsize',12,'FontName','Times New Roman')
xlabel('Sample','fontsize',12,'FontName','Times New Roman')
xlim([1 M]);
%-------------------------------------------------------------------------------------
figure
subplot(2,1,1)
plot(T_sim2,'-s','Color',[0 0 255]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
hold on 
plot(T_test,'-o','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[0 0 0]./255)
legend('LSTM prediciting the testing data','True analytical data','Location','NorthWest','FontName','Times New Roman');
title('LSTM predicited testing data and the true value','fontsize',12,'FontName','Times New Roman')
xlabel('Sample','fontsize',12,'FontName','Times New Roman');
ylabel('Data','fontsize',12,'FontName','Times New Roman');
xlim([1 N])
%-------------------------------------------------------------------------------------
subplot(2,1,2)
bar((T_sim2 - T_test )./T_test)   
legend('The relative error of LSTM model testing set','Location','NorthEast','FontName','Times New Roman')
title('The relative error of LSTM model testing set','fontsize',12,'FontName','Times New Roman')
ylabel('Error','fontsize',12,'FontName','Times New Roman')
xlabel('Sample','fontsize',12,'FontName','Times New Roman')
xlim([1 N]);

%% 12.预测未来
P = 88;% 预测未来数量
YPred_3 = [];%预测结果清零 
[T_sim3] = yuceweilai(net,XTrain,data,P,YPred_3,sig,mu)

%%  13.绘图
figure
plot(1:size(data,1),data,'-s','Color',[255 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[250 0 0]./255)
hold on 
plot(size(data,1)+1:size(data,1)+P,T_sim3,'-o','Color',[150 150 150]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[150 150 150]./255)
legend( 'The prediciting result of LSTM model','Location','NorthWest','FontName','Times New Roman');
title('The prediciting result of LSTM model','fontsize',12,'FontName','Times New Roman')
xlabel('Sample','fontsize',12,'FontName','Times New Roman');
ylabel('Data','fontsize',12,'FontName','Times New Roman');
