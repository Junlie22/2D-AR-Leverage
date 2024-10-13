function A = LARTVAD(data,Db,beta,lambda)
mu = 0.01;
data = (data-min(data(:)))/(max(data(:))-min(data(:)));
dim = size(data);

N       = size(Db,2);


%% D1 and D2
DD = cell(1,2);     
for i = 1:2
diaga   = ones(dim(i),1);  
diagb   = ones(dim(i)-1,1);
DD{i}   = diag(-diaga)+diag(diagb,1);
DD{i}(end,1) = 1;
end
D1 = DD{1};
D2 = DD{2};

d1         = zeros(dim(1),dim(2));
d1(end,1)  = 1;  d1(1,1) = -1;
d2         = zeros(dim(1),dim(2));
d2(1,end)  = 1;  d2(1,1) = -1;
fd1      = fft2(d1);
fd2      = fft2(d2);
Sig = ((abs(fd1)).^2+(abs(fd2)).^2);
Sig = Sig(:);

%% initialization
dimB = [dim(1),dim(2),N];
B = randn(dimB);
A  = zeros(dim);
P1 = zeros(dimB);
P2 = zeros(dimB);
P3 = zeros(dimB);
P4 = zeros(dim);
Y  = data;
BD1 = double(ttm(tensor(B),D1,1));
BD2 = double(ttm(tensor(B),D2,2));
[U2,E2,~]  = svd(Db'*Db);
Sig2       = diag(E2);
T           = repmat(Sig2',dimB(1)*dimB(2),1)+repmat(Sig,1,dimB(3)) + 1;
T           = 1./T;

for i = 1:100
    LastB = B;
    Z = prox_tnn(B + P3/mu,1/mu);%update Z
    S1 = prox_l1(BD1 + P1/mu,lambda/mu);%update S1
    S2 = prox_l1(BD2 + P2/mu,lambda/mu);%update S2
    A = Thres_21(Y - double(ttm(tensor(B),Db,3))+P4/mu,beta/mu);%update A
    
    %update B   
    K          = double(ttm(tensor(Y-A),Db',3))+Z+(double(ttm(tensor(S1-P1/mu),D1',1))...
                +double(ttm(tensor(S2-P2/mu),D2',2)))+double(ttm(tensor(P4),Db',3))/mu-P3/mu;
    K3         = Unfold(K,dimB,3);
    temp       = T.*(calF(K3',dimB(1),dimB(2))*U2);
    B3t        = real(calFt(temp,dimB(1),dimB(2)))*U2';
    B          = Fold(B3t',dimB,3);                                      
    
    % update Lagrange multipliers
    BD1 = double(ttm(tensor(B),D1,1));
    BD2 = double(ttm(tensor(B),D2,2));
    P1 = P1 + mu*(BD1-S1);
    P2 = P2 + mu*(BD2-S2);
    P3 = P3 + mu*(B-Z);
    P4 = P4 + mu*(Y - double(ttm(tensor(B),Db,3))-A);
    mu = min(1.2*mu,1e10);
    
    error(i) = norm(LastB(:)-B(:));
    if error(i) < 0.00001
        break;
    end
end

