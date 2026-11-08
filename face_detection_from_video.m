clc; clear; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
resultDir = fullfile(scriptDir, 'results');
if ~exist(resultDir, 'dir'), mkdir(resultDir); end

reader = VideoReader(fullfile(projectDir, 'deneme1.mp4'));
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');

% Her Viola-Jones adayi icinde aranacak ek yuz kanitlari.
eyeBig = vision.CascadeObjectDetector('EyePairBig');
eyeBig.UseROI = true;
eyeSmall = vision.CascadeObjectDetector('EyePairSmall');
eyeSmall.UseROI = true;
noseDetector = vision.CascadeObjectDetector('Nose');
noseDetector.UseROI = true;

iouThr = 0.30;
sizeThr = 0.60;
minHits = 5;
minEvidence = 2;  % Bes eslesmenin en az ikisinde goz/burun/ten kaniti
maxMiss = 3;

tracks = struct('bbox',{}, 'hits',{}, 'evidence',{}, ...
    'miss',{}, 'confirmed',{});
rawCount = [];
filteredCount = [];
frameNo = 1;

rawPath = fullfile(resultDir, 'task2_baseline.avi');
filteredPath = fullfile(resultDir, 'task2_improved.avi');
rawWriter = VideoWriter(rawPath);
filteredWriter = VideoWriter(filteredPath);
rawWriter.FrameRate = reader.FrameRate;
filteredWriter.FrameRate = reader.FrameRate;
rawWriter.Quality = 95;
filteredWriter.Quality = 95;
open(rawWriter); open(filteredWriter);

figure('Position',[100 100 1200 450]);

while hasFrame(reader)
    frame = readFrame(reader);
    gray = rgb2gray(frame);
    dets = double(faceDetector(gray));

    % Her aday goz, burun ve ten bilgisiyle ayri puanlanir.
    hasEvidence = false(size(dets,1),1);
    for d = 1:size(dets,1)
        hasEvidence(d) = validateFace(frame, gray, dets(d,:), ...
            eyeBig, eyeSmall, noseDetector);
    end

    nD = size(dets,1);
    nT = numel(tracks);
    matchedD = false(nD,1);
    matchedT = false(1,nT);

    % Eski koddaki IoU ve boyut benzerligiyle bire bir eslestirme.
    if nD > 0 && nT > 0
        M = zeros(nD,nT);
        for d = 1:nD
            for t = 1:nT
                M(d,t) = iouScore(dets(d,:), tracks(t).bbox);
            end
        end

        while true
            [bestIoU,k] = max(M(:));
            if bestIoU < iouThr, break; end
            [d,t] = ind2sub(size(M),k);

            widthRatio = min(dets(d,3),tracks(t).bbox(3)) / ...
                max(dets(d,3),tracks(t).bbox(3));
            heightRatio = min(dets(d,4),tracks(t).bbox(4)) / ...
                max(dets(d,4),tracks(t).bbox(4));
            if widthRatio < sizeThr || heightRatio < sizeThr
                M(d,t) = 0;
                continue;
            end

            tracks(t).bbox = 0.5*tracks(t).bbox + 0.5*dets(d,:);
            tracks(t).hits = tracks(t).hits + 1;
            tracks(t).evidence = tracks(t).evidence + hasEvidence(d);
            tracks(t).miss = 0;
            tracks(t).confirmed = tracks(t).hits >= minHits && ...
                tracks(t).evidence >= minEvidence;
            matchedD(d) = true;
            matchedT(t) = true;
            M(d,:) = 0;
            M(:,t) = 0;
        end
    end

    for t = find(~matchedT)
        tracks(t).miss = tracks(t).miss + 1;
    end
    if ~isempty(tracks)
        remove = (~[tracks.confirmed] & [tracks.miss] >= 1) | ...
                 ([tracks.confirmed] & [tracks.miss] > maxMiss);
        tracks(remove) = [];
    end

    for d = find(~matchedD)'
        tracks(end+1) = struct('bbox',dets(d,:), 'hits',1, ...
            'evidence',double(hasEvidence(d)), 'miss',0, ...
            'confirmed',false); %#ok<SAGROW>
    end

    confirmedBoxes = zeros(0,4);
    for t = 1:numel(tracks)
        if tracks(t).confirmed
            confirmedBoxes(end+1,:) = ...
                tightenSides(frame,tracks(t).bbox); %#ok<SAGROW>
        end
    end

    rawFrame = frame;
    filteredFrame = frame;
    if ~isempty(dets)
        rawFrame = insertShape(rawFrame,'Rectangle',dets, ...
            'Color','red','LineWidth',5);
    end
    if ~isempty(confirmedBoxes)
        filteredFrame = insertShape(filteredFrame,'Rectangle', ...
            confirmedBoxes,'Color','green','LineWidth',5);
    end

    writeVideo(rawWriter,rawFrame);
    writeVideo(filteredWriter,filteredFrame);

    if mod(frameNo,20) == 0
        subplot(1,2,1); imshow(rawFrame); title('Ham Viola-Jones');
        subplot(1,2,2); imshow(filteredFrame);
        title('Goz + burun + ten + IoU'); drawnow;
    end

    rawCount(frameNo) = nD;
    filteredCount(frameNo) = size(confirmedBoxes,1);
    frameNo = frameNo + 1;
end

close(rawWriter); close(filteredWriter);

figure;
plot(rawCount,'r'); hold on; plot(filteredCount,'g','LineWidth',2);
legend('Ham Viola-Jones','Dogrulanmis'); grid on;
xlabel('Kare no'); ylabel('Tespit sayisi');

fprintf('Toplam kare       : %d\n',numel(rawCount));
fprintf('Ham tespit        : %d\n',sum(rawCount));
fprintf('Dogrulanmis tespit: %d\n',sum(filteredCount));
fprintf('Ham video         : %s\n',rawPath);
fprintf('Dogrulanmis video : %s\n',filteredPath);

% ---------------- Yardimci fonksiyonlar ----------------

function valid = validateFace(rgb,gray,box,eyeBig,eyeSmall,nose)
    imageSize = size(gray);
    x = box(1); y = box(2); w = box(3); h = box(4);
    eyeROI = clipROI([x,y+0.08*h,w,0.55*h],imageSize);
    noseROI = clipROI([x+0.15*w,y+0.30*h,0.70*w,0.58*h],imageSize);

    eyes = zeros(0,4);
    if eyeROI(3) >= 45 && eyeROI(4) >= 11
        eyes = eyeBig(gray,eyeROI);
    end
    if isempty(eyes) && eyeROI(3) >= 22 && eyeROI(4) >= 5
        eyes = eyeSmall(gray,eyeROI);
    end

    noses = zeros(0,4);
    if noseROI(3) >= 18 && noseROI(4) >= 15
        noses = nose(gray,noseROI);
    end

    faceROI = clipROI(box,imageSize);
    x1 = faceROI(1); y1 = faceROI(2);
    crop = rgb(y1:y1+faceROI(4)-1, x1:x1+faceROI(3)-1, :);
    color = rgb2ycbcr(crop);
    Cb = color(:,:,2); Cr = color(:,:,3);
    skinRatio = nnz((Cb>=77 & Cb<=127) & (Cr>=133 & Cr<=173)) / ...
        (size(Cb,1)*size(Cb,2));

    score = 2*~isempty(eyes) + ~isempty(noses) + ...
        (skinRatio >= 0.12 && skinRatio <= 0.85);
    valid = score >= 3;
end


function roi = clipROI(box,imageSize)
    x1 = max(1,floor(box(1))); y1 = max(1,floor(box(2)));
    x2 = min(imageSize(2),ceil(box(1)+box(3)-1));
    y2 = min(imageSize(1),ceil(box(2)+box(4)-1));
    roi = [x1,y1,max(1,x2-x1+1),max(1,y2-y1+1)];
end


function box = tightenSides(img,box)
    % Ten bolgesinin sol ve sag sinirini bul; ust ve alt kenari degistirme.
    roi = clipROI(box,size(img(:,:,1)));
    x = roi(1); y = roi(2); w = roi(3); h = roi(4);
    crop = img(y:y+h-1,x:x+w-1,:);
    color = rgb2ycbcr(crop);
    Cb = color(:,:,2); Cr = color(:,:,3);
    skin = (Cb>=77 & Cb<=127) & (Cr>=133 & Cr<=173);
    skin = imclose(skin,strel('disk',5,0));
    cc = bwconncomp(skin);
    if cc.NumObjects == 0, return; end

    [~,big] = max(cellfun(@numel,cc.PixelIdxList));
    mask = false(size(skin)); mask(cc.PixelIdxList{big}) = true;
    cols = find(any(mask,1));
    if isempty(cols), return; end

    margin = round(0.04*box(3));
    left = max(1,x+cols(1)-1-margin);
    right = min(size(img,2),x+cols(end)-1+margin);
    width = right-left+1;
    if width >= 0.65*box(3) && width <= 1.05*box(3)
        box(1) = left; box(3) = width;
    end
end


function score = iouScore(a,b)
    iw = max(0,min(a(1)+a(3),b(1)+b(3))-max(a(1),b(1)));
    ih = max(0,min(a(2)+a(4),b(2)+b(4))-max(a(2),b(2)));
    intersection = iw*ih;
    union = a(3)*a(4)+b(3)*b(4)-intersection;
    if union > 0, score = intersection/union; else, score = 0; end
end
