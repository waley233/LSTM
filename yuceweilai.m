function [T_sim3] = yuceweilai(net,XTrain,data,P,YPred_3,sig,mu)
net1 = resetState(net);
net1 = predictAndUpdateState(net1,XTrain);
[net1,YPred_3] = predictAndUpdateState(net1,data(end));
for i = 2:P
    [net1,YPred_3(:,i)] = predictAndUpdateState(net1,YPred_3(:,i-1),'ExecutionEnvironment','cpu');
end
T_sim3 = sig*YPred_3 + mu;
end

