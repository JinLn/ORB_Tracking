clc
clear
close all

%% Step 1: ��ȡ�궨������ͼ��
data = load('calibrationSession.mat');
% BF=����*����
% Z=BF/(ul-ur)
BF = data.calibrationSession.CameraParameters.CameraParameters1.Intrinsics.FocalLength(1,1) * -data.calibrationSession.CameraParameters.TranslationOfCamera2(1,1)/1000; 

leftimgpath = '.\leftimg\';
rightimgpath = '.\rightimg\';
ext = '*.png';
leftimgdir = dir([leftimgpath ext]);
rightimgdir = dir([rightimgpath ext]);
leftimgname = {leftimgdir.name};
rightimgname = {rightimgdir.name};

%% Step 2: ѭ��ͼƬ
Flag=true; 
bigin = 1; % �ڼ��ſ�ʼ
for ni = bigin:2:length(leftimgname)
    leftpathname = [leftimgpath leftimgname{ni}];
    rightpathname = [rightimgpath rightimgname{ni}];
    leftImage = imread(leftpathname);
    rightImage = imread(rightpathname);

    %% Step 2.1: ѡ��Ŀ��������������ͼ���п�ѡһ������
    if ni ==1 || Flag
        Flag=false;
        imshow(leftImage),title('Draw a tracking box');
        k = waitforbuttonpress; % �ȴ���갴��
        point1 = get(gca,'CurrentPoint'); % ��갴����
        finalRect = rbbox; %
        point2 = get(gca,'CurrentPoint'); % ����ɿ���
        point1 = point1(1,1:2); % ��ȡ��������
        point2 = point2(1,1:2);
        p1 = min(floor(point1),floor(point2)); % ����λ��
        p2 = max(floor(point1),floor(point2));
        offset = abs(floor(point1)-floor(point2)); % offset(1)��ʾ��offset(2)��ʾ��
        x = [p1(1) p1(1)+offset(1) p1(1)+offset(1) p1(1) p1(1)];
        y = [p1(2) p1(2) p1(2)+offset(2) p1(2)+offset(2) p1(2)];
        hold on %��ֹplotʱ��˸
        plot(x,y,'r','LineWidth',3);
    end

    %% Step 2.2: ��ȡORB������
    leftPoints = detectORBFeatures(leftImage,'ScaleFactor',1.1,'NumLevels',8, 'ROI',[p1(1,1) p1(1,2) max(p2(1,1)-p1(1,1),64) max(p2(1,2)-p1(1,2),64)]);
    rightPoints = detectORBFeatures(rightImage,'ScaleFactor',1.1,'NumLevels',8, 'ROI',[p1(1,1)-60 p1(1,2)-60 p2(1,1)-p1(1,1)+60 p2(1,2)-p1(1,2)+60]);
    if leftPoints.Count < 3 || rightPoints.Count < 3
        Flag=true;
        continue;
    end

    %% Step 2.3: ��������������ͼ�����������
    [leftFeatures, leftPoints] = extractFeatures(leftImage, leftPoints);
    [rightFeatures, rightPoints] = extractFeatures(rightImage, rightPoints);

    %% Step 2.4: ��������һ��ƥ���
    matchPairs = matchFeatures(leftFeatures, rightFeatures,'MatchThreshold',100,'MaxRatio',0.6);
    
    %% Step 2.5: �������
    depthZ=[];
    for i=size(matchPairs):-1:1
         d = BF / (leftPoints.Location(matchPairs(i,1),1) - rightPoints.Location(matchPairs(i,2),1));
        if abs(d-depthZ) > 1
            matchPairs(i,:)=[];% ˫Ŀƥ��Ķ�Ӧ��ϵ,�޳����
        end
            depthZ(end+1,1) = d;
    end
    if size(depthZ)==0
        Flag=true;
        continue;
    end
    meanDepthZ=mean2(depthZ);
     %% Step 2.6: ��ʾ����
    leftImageCopy = leftImage;
    leftImageCopy(:,:,2) = leftImage;
    leftImageCopy(:,:,3) = leftImage;
    textlocation=[(p2(1,1)+p1(1,1))/2 (p2(1,2)+p1(1,2))/2];
    leftImageCopy = insertText(leftImageCopy,textlocation,num2str(meanDepthZ),'FontSize',40,'AnchorPoint','CenterBottom');
    figure(1);
    imshow(leftImageCopy),title('Distance(m)');


    %% Step 2.7: ��ʾ˫Ŀƥ���
    %show
    matchedLeftPoints = leftPoints(matchPairs(:, 1), :);
    matchedRightPoints = rightPoints(matchPairs(:, 2), :);
    figure(2);
    hold on
    pt=[ceil(p1(1,1)),ceil(p1(1,2))];
    wSize=[ ceil(p2(1,1)-p1(1,1)),ceil(p2(1,2)-p1(1,2))];
    leftImage = drawRect(leftImage,pt,wSize,4 ); %  drawRect����ֱ�ӵ��ã��� drawRect.m
    showMatchedFeatures(leftImage, rightImage, matchedLeftPoints,matchedRightPoints, 'montage');
    title('Binocular Matched Points (Including Outliers)');
    
    %% Step 2.8: ǰ��֡ƥ��
    if ni ~= bigin % �ڶ�֡��ʼ
        %% Step 2.8.1: ǰ��֡������ƥ��
        matchPairsLastandCur = matchFeatures(lastleftFeatures, leftFeatures,'Method','Approximate','MatchThreshold',40,'MaxRatio',0.4); % ���е� 'Exhaustive'(Ĭ��)'Approximate'
        matchedlastPoints = lastleftPoint(matchPairsLastandCur(:, 1), :);
        matchedCurPoints = leftPoints(matchPairsLastandCur(:, 2), :);
        if size(matchedLeftPoints)==0
            Flag=true;
            continue;
        elseif size(matchedLeftPoints) > 10
            %[tform, matchedlastPoints, matchedCurPoints] = estimateGeometricTransform(matchedlastPoints, matchedCurPoints, 'projective'); % 'similarity' | 'affine' | 'projective'
        end
        
        %% Step 2.8.2: ��ʾǰ��֡ƥ��
        figure(3);
        hold on;
        showMatchedFeatures(lastleftImage, leftImage, matchedlastPoints,matchedCurPoints, 'falsecolor'); % falsecolor (default) | blend | montage
        title('Last and Current Image Matched Points');
        
        %% Step 2.8.3: ���±߿�
        if matchedCurPoints.Count > 10
            p1(1,1) = min(matchedCurPoints.Location(:,1)); 
            p2(1,1) = max(matchedCurPoints.Location(:,1)); 
            p1(1,2) = min(matchedCurPoints.Location(:,2)); 
            p2(1,2) = max(matchedCurPoints.Location(:,2)); 

            if p2(1,1) - p1(1,1) < offset(1)/2 && p2(1,2) - p1(1,2) < offset(2)/2
                % offset(1)��ʾ��offset(2)��ʾ��
                p1(1,1) = (max(matchedCurPoints.Location(:,1)) + min(matchedCurPoints.Location(:,1)))/2 - offset(1)/2;
                p2(1,1) = (max(matchedCurPoints.Location(:,1)) + min(matchedCurPoints.Location(:,1)))/2 + offset(1)/2;
                p1(1,2) = (max(matchedCurPoints.Location(:,2)) + min(matchedCurPoints.Location(:,2)))/2 - offset(2)/2; 
                p2(1,2) = (max(matchedCurPoints.Location(:,2)) + min(matchedCurPoints.Location(:,2)))/2 + offset(2)/2; 
            end
        end
        
    end
    
    %% Step 2.9: ������һ֡
    lastleftImage = leftImage;
    lastleftPoint = leftPoints;
    lastleftFeatures = leftFeatures;
    
    if waitforbuttonpress % ��������ߵ����һ֡
    end
end