function makeDb(trFileLoc,imLoc,svLoc,height,width,alpha,boxsz,cropDivN,minFrameNo)

%initiating required variables
no_ang = 180/alpha;
histmat = [];
actmat = {};
trNo = 0;

%folder that contains training videos
folders = dir(trFileLoc);
for folderInd=1:length(folders)
    act = folders(folderInd).name;
    if strcmp(act,'.') || strcmp(act,'..')
        continue
    end
    
    actFileLoc = strcat(trFileLoc,act);
    vids = dir(actFileLoc);
    for vidInd=1:length(vids)
        vidName = vids(vidInd).name;
        if strcmp(vidName,'.') || strcmp(vidName,'..')
            continue
        end
        
        %creating video-reader,foreground-detector and blob-detector
        %objects
        videoFReader = vision.VideoFileReader(strcat(strcat(actFileLoc,'\'),vidName));
        detector = vision.ForegroundDetector('NumTrainingFrames',10,'NumGaussians',5);
        blob = vision.BlobAnalysis(...
            'CentroidOutputPort', false, 'AreaOutputPort', false, ...
            'BoundingBoxOutputPort', true, ...
            'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 250);
        
        frameNo = 0;
        while ~isDone(videoFReader)
            frame = step(videoFReader);
            frameNo = frameNo+1;
            
            fgMask = step(detector,frame);
            box = step(blob,fgMask);
            
            %cond1 - current frame number is greater than threshold frame number. This condition is kept too ensure that initial not good foreground masks are removed.
            %cond2 - One bounding box returns 1*4 matrix. Hence it checks if one bounding exisits in the mask
            cond1 = (frameNo >= minFrameNo);
            cond2 = (size(box,1) == 1 && size(box,2) == 4);
            
            if cond1 && cond2
                %cropping the fgMask to the bounding box
                cropped = cropToBoundingBox(fgMask,box);
                %resizing the bounding box image to a boxsz*x or x*boxsz dimension where x<=boxsz
                cropped = resizeToBoxsize(cropped,boxsz);
                
                cropht = size(cropped,1);cropwt = size(cropped,2);
                nwcropht = floor(cropht/cropDivN);
                nwcropwt = floor(cropwt/cropDivN);
                histframe = [];
                %looping through each division in cropped bounding box and
                %extracting feature information for training
                for xi=1:cropDivN
                    for yi=1:cropDivN
                        newcrop = cropped((nwcropht*(xi-1))+1:nwcropht*xi,(nwcropwt*(yi-1))+1:nwcropwt*yi);
                        histarr = zeros(1,no_ang);
                        i=1;
                        for ang=0:alpha:180-alpha
                            temp = load(sprintf('%sconv_%d_%d_%d.mat',imLoc,height,width,ang));
                            temp = temp.temp;
                            thres = sum(temp(:)==1);
                            
                            if size(temp,1)>size(newcrop,1) || size(temp,2)>size(newcrop,2)
                                continue
                            end
                            
                            %convolution of the template(temp) on the image(newcrop).The convoluted image is then cropped to the size of image(newcrop)
                            convmat = convolutionCustom(newcrop,temp);
                            
                            %The histogram corresponding to 'i'th angle(ang) is
                            %equated to the number of elements in convolution
                            %matrix(convmat) that is greater than threshold(thres)
                            histarr(i) = sum(convmat(:)>=thres);
                            i = i+1;
                        end
                        histframe = [histframe,histarr];
                    end
                end
                histmat = [histmat;histframe];
                trNo = trNo+1;
                actmat{trNo} = act;
            end
        end
        release(videoFReader);
    end
end

%saving extracted features and corresponding action
save(sprintf('%sX.mat',svLoc),'histmat');
save(sprintf('%sY.mat',svLoc),'actmat');