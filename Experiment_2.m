clc
clear
close all

%% Step 1: 读取标定参数和图像
data = load('calibrationSession.mat');
% BF=基线*焦距
% Z=BF/(ul-ur)
BF = data.calibrationSession.CameraParameters.CameraParameters1.Intrinsics.FocalLength(1,1) * -data.calibrationSession.CameraParameters.TranslationOfCamera2(1,1)/1000; 

leftimgpath = '.\leftimg\';
rightimgpath = '.\rightimg\';
ext = '*.png';
leftimgdir = dir([leftimgpath ext]);
rightimgdir = dir([rightimgpath ext]);
leftimgname = {leftimgdir.name};
rightimgname = {rightimgdir.name};

%% Step 2: 循环图片
Flag=true; 
bigin = 1; % 第几张开始
for ni = bigin:2:length(leftimgname)
    leftpathname = [leftimgpath leftimgname{ni}];
    rightpathname = [rightimgpath rightimgname{ni}];
    leftImage = imread(leftpathname);
    rightImage = imread(rightpathname);

    %% Step 2.1: 选择目标矩形区域，鼠标在图像中框选一个区域
    if ni ==1 || Flag
        Flag=false;
        imshow(leftImage),title('Draw a tracking box');
        k = waitforbuttonpress; % 等待鼠标按下
        point1 = get(gca,'CurrentPoint'); % 鼠标按下了
        finalRect = rbbox; %
        point2 = get(gca,'CurrentPoint'); % 鼠标松开了
        point1 = point1(1,1:2); % 提取出两个点
        point2 = point2(1,1:2);
        p1 = min(floor(point1),floor(point2)); % 计算位置
        p2 = max(floor(point1),floor(point2));
        offset = abs(floor(point1)-floor(point2)); % offset(1)表示宽，offset(2)表示高
        x = [p1(1) p1(1)+offset(1) p1(1)+offset(1) p1(1) p1(1)];
        y = [p1(2) p1(2) p1(2)+offset(2) p1(2)+offset(2) p1(2)];
        hold on %防止plot时闪烁
        plot(x,y,'r','LineWidth',3);
    end

    %% Step 2.2: 提取ORB特征点
    leftPoints = detectORBFeatures(leftImage,'ScaleFactor',1.1,'NumLevels',8, 'ROI',[p1(1,1) p1(1,2) max(p2(1,1)-p1(1,1),64) max(p2(1,2)-p1(1,2),64)]);
    rightPoints = detectORBFeatures(rightImage,'ScaleFactor',1.1,'NumLevels',8, 'ROI',[p1(1,1)-60 p1(1,2)-60 p2(1,1)-p1(1,1)+60 p2(1,2)-p1(1,2)+60]);
    if leftPoints.Count < 3 || rightPoints.Count < 3
        Flag=true;
        continue;
    end

    %% Step 2.3: 根据特征点生成图像的特征向量
    [leftFeatures, leftPoints] = extractFeatures(leftImage, leftPoints);
    [rightFeatures, rightPoints] = extractFeatures(rightImage, rightPoints);

    %% Step 2.4: 初步建立一个匹配对
    matchPairs = matchFeatures(leftFeatures, rightFeatures,'MatchThreshold',100,'MaxRatio',0.6);
    
    %% Step 2.5: 计算距离
    depthZ=[];
    for i=size(matchPairs):-1:1
         d = BF / (leftPoints.Location(matchPairs(i,1),1) - rightPoints.Location(matchPairs(i,2),1));
        if abs(d-depthZ) > 1
            matchPairs(i,:)=[];% 双目匹配的对应关系,剔除外点
        end
            depthZ(end+1,1) = d;
    end
    if size(depthZ)==0
        Flag=true;
        continue;
    end
    meanDepthZ=mean2(depthZ);
     %% Step 2.6: 显示距离
    leftImageCopy = leftImage;
    leftImageCopy(:,:,2) = leftImage;
    leftImageCopy(:,:,3) = leftImage;
    textlocation=[(p2(1,1)+p1(1,1))/2 (p2(1,2)+p1(1,2))/2];
    leftImageCopy = insertText(leftImageCopy,textlocation,num2str(meanDepthZ),'FontSize',40,'AnchorPoint','CenterBottom');
    figure(1);
    imshow(leftImageCopy),title('Distance(m)');


    %% Step 2.7: 显示双目匹配点
    %show
    matchedLeftPoints = leftPoints(matchPairs(:, 1), :);
    matchedRightPoints = rightPoints(matchPairs(:, 2), :);
    figure(2);
    hold on
    pt=[ceil(p1(1,1)),ceil(p1(1,2))];
    wSize=[ ceil(p2(1,1)-p1(1,1)),ceil(p2(1,2)-p1(1,2))];
    leftImage = drawRect(leftImage,pt,wSize,4 ); %  drawRect函数直接调用，见 drawRect.m
    showMatchedFeatures(leftImage, rightImage, matchedLeftPoints,matchedRightPoints, 'montage');
    title('Binocular Matched Points (Including Outliers)');
    
    %% Step 2.8: 前后帧匹配
    if ni ~= bigin % 第二帧开始
        %% Step 2.8.1: 前后帧特征点匹配
        matchPairsLastandCur = matchFeatures(lastleftFeatures, leftFeatures,'Method','Approximate','MatchThreshold',40,'MaxRatio',0.4); % 所有点 'Exhaustive'(默认)'Approximate'
        matchedlastPoints = lastleftPoint(matchPairsLastandCur(:, 1), :);
        matchedCurPoints = leftPoints(matchPairsLastandCur(:, 2), :);
        if size(matchedLeftPoints)==0
            Flag=true;
            continue;
        elseif size(matchedLeftPoints) > 10
            %[tform, matchedlastPoints, matchedCurPoints] = estimateGeometricTransform(matchedlastPoints, matchedCurPoints, 'projective'); % 'similarity' | 'affine' | 'projective'
        end
        
        %% Step 2.8.2: 显示前后帧匹配
        figure(3);
        hold on;
        showMatchedFeatures(lastleftImage, leftImage, matchedlastPoints,matchedCurPoints, 'falsecolor'); % falsecolor (default) | blend | montage
        title('Last and Current Image Matched Points');
        
        %% Step 2.8.3: 更新边框
        if matchedCurPoints.Count > 10
            p1(1,1) = min(matchedCurPoints.Location(:,1)); 
            p2(1,1) = max(matchedCurPoints.Location(:,1)); 
            p1(1,2) = min(matchedCurPoints.Location(:,2)); 
            p2(1,2) = max(matchedCurPoints.Location(:,2)); 

            if p2(1,1) - p1(1,1) < offset(1)/2 && p2(1,2) - p1(1,2) < offset(2)/2
                % offset(1)表示宽，offset(2)表示高
                p1(1,1) = (max(matchedCurPoints.Location(:,1)) + min(matchedCurPoints.Location(:,1)))/2 - offset(1)/2;
                p2(1,1) = (max(matchedCurPoints.Location(:,1)) + min(matchedCurPoints.Location(:,1)))/2 + offset(1)/2;
                p1(1,2) = (max(matchedCurPoints.Location(:,2)) + min(matchedCurPoints.Location(:,2)))/2 - offset(2)/2; 
                p2(1,2) = (max(matchedCurPoints.Location(:,2)) + min(matchedCurPoints.Location(:,2)))/2 + offset(2)/2; 
            end
        end
        
    end
    
    %% Step 2.9: 更新上一帧
    lastleftImage = leftImage;
    lastleftPoint = leftPoints;
    lastleftFeatures = leftFeatures;
    
    if waitforbuttonpress % 任意键或者点击下一帧
    end
end