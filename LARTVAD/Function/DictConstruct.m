function Dic = DictConstruct(data,K)
% superpixel segment based on kmeans

data    = (data-min(data(:))) / (max(data(:))-min(data(:)));
dim     = size(data);
s       = ceil(sqrt(dim(1)*dim(2)/K));
MatData = Unfold(data,dim,3);
errTh   = 10^-2;
wDs     = 0.5^2;
Mdata   = Unfold(data,dim,3);
[U,S,V] = svd(MatData,'econ');
FPCA = U(:,1)*S(1,1)*V(:,1)';
data = Fold(FPCA,dim,3);

m = size(data,1);
n = size(data,2);
 
h = floor(m/s);
w = floor(n/s);
rowR = floor((m-h*s)/2);
colR = floor((n-w*s)/2);
rowStart = (rowR + 1):s:(m-s+1);
rowStart(1) = 1;
rowEnd = rowStart + s;
rowEnd(1) = rowR + s;
rowEnd(end) = m;
colStart = (colR+1):s:(n-s+1);
colStart(1) = 1;
colEnd = colStart+s;
colEnd(1) = colR+s;
colEnd(end) = n;
rowC = floor((rowStart + rowEnd - 1)/2);
colC = floor((colStart + colEnd - 1)/2);

temp=zeros(m,n);
temp(rowStart,:)=1;
temp(:,colStart)=1;
for i=1:h
    for j=1:w
        temp(rowC(i),colC(j))=1;
    end
end
 

Y = data(:,:,1);
data = (Y-min(Y(:)))/(max(Y(:))-min(Y(:)));
Y = data;

 
f1 = fspecial('sobel');
f2 = f1';
gx = imfilter(Y,f1);
gy = imfilter(Y,f2);
G  = sqrt(gx.^2+gy.^2); 
 

rowC_std = repmat(rowC',[1,w]);
colC_std = repmat(colC,[h,1]);
rowC=rowC_std;
colC=colC_std;
for i=1:h
    for j=1:w
        block=G(rowC(i,j)-1:rowC(i,j)+1,colC(i,j)-1:colC(i,j)+1);
        [minVal,idxArr]=min(block(:));
        jOffset=floor((idxArr(1) + 2)/3);
        iOffset=idxArr(1)-3*(jOffset-1);
        rowC(i,j)=rowC(i,j)+iOffset;
        colC(i,j)=colC(i,j)+jOffset;
    end
end
 
%% KMeans
Label = zeros(m,n)-1;
dis = Inf*ones(m,n);
M = reshape(data,m*n,size(data,3)); 

colorC=zeros(h,w,size(data,3));
for i=1:h
    for j=1:w
        colorC(i,j,:)=data(rowC(i),colC(j),:);
    end
end
uniMat = cat(3,colorC,rowC,colC); 
uniMat = reshape(uniMat,h*w,size(data,3)+2);
iter=1;
while(1)
    uniMat_old = uniMat;
    for k=1:h*w
        c           = floor((k-1)/h) + 1;
        r           = k - h*(c-1);
        rowCidx     = rowC(r,c);
        colCidx     = colC(r,c); 

        rowStart    = max(1,rowC_std(r,c)-s);
        rowEnd      = min(m,rowC_std(r,c)+s-1);
        colStart    = max(1,colC_std(r,c)-s);
        colEnd      = min(n,colC_std(r,c)+s);
        colorC      = M((colCidx-1)*m + rowCidx,:); % current centers
        for i = rowStart:rowEnd
            for j = colStart:colEnd
                colorCur = M((j-1)*m+i,:); 
                dc = norm(colorC-colorCur);
                ds = norm([i-rowCidx,j-colCidx]);
                d = dc^2 + (ds/s)^2;
                if d<dis(i,j)
                    dis(i,j)=d;
                    Label(i,j)=k;
                end
            end
        end
    end
    iter=iter+1; 
    % update centers
    colorC = zeros(h,w,size(data,3));
    for k=1:h*w
        num = 0;
        sumColor = zeros(1,size(data,3));    
        sumR = 0;
        sumC = 0;
        c = floor((k-1)/h)+1;
        r = k-h*(c-1);
        rowCidx = rowC_std(r,c);
        colCidx = colC_std(r,c);
        rowStart=max(1,rowCidx-s);
        rowEnd=min(m,rowCidx+s-1);
        colStart=max(1,colCidx-s);
        colEnd=min(n,colCidx+s);
        
        for row=rowStart:rowEnd
            for col=colStart:colEnd
                if Label(row,col)==k
                    num = num + 1;
                    sumR = sumR + row;
                    sumC = sumC + col;
                    color = reshape(data(row,col,:),1,size(data,3));
                    sumColor = sumColor + color;
                end
            end
        end
        colorC(r,c,:) = sumColor/num;
        rowC(r,c) = round(sumR/num);
        colC(r,c) = round(sumC/num);
    end
    uniMat=cat(3,colorC,rowC,colC);
    uniMat=reshape(uniMat,h*w,size(data,3)+2);
    diff = uniMat-uniMat_old;
    diff(:,1:2) = sqrt(wDs)*diff(:,1:2)/s;
    err = norm(diff)/sqrt(h*w);
    if err < errTh 
        break;
    end
end


% select an atom from each cluster
class = {};
for k=1:max(Label(:))  
    class{k} = Mdata(:,Label==k);
end
 

Dic = [];
for i = 1:length(class)
    datai = class{i};
    N1 = size(datai,2);
    if N1 < 1
        continue;
    end
    [U,~,~] = svd(datai,'econ');
    S = U(:,1)*U(:,1)';   
    disi = zeros(N1,1);
    for j = 1:N1
        disi(j) = norm(datai(:,j)-S*datai(:,j));
    end
    [~,IND]     = sort(disi,'ascend');
    ind      = IND(1);
    Dic = [Dic datai(:,ind) ];
end

