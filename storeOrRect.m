function storeOrRect(height,width,alpha,fileLocation) 
%The program stores the oriented rectangles of specified height and width as mat files
%
%height - height of the rectangle
%width - width of the rectangle
%alpha - orientation change
%fileLocation - location of the file where the oriented rectangles have to be stored

%rect - rectangle of height*width
rect = ones(height,width);
for ang=0:alpha:180-alpha
    
    %rotating the rectangle to required(ang) orientation
    temp = imrotate(rect,ang);
    
    %4 while loops remove padded zeros on 4 sides
    while size(temp,1) > 0
        if(sum(temp(1,:)) == 0)
            temp = temp(2:end,:);
        else
            break;
        end
    end
    while size(temp,1) > 0
        if(sum(temp(end,:)) == 0)
            temp = temp(1:end-1,:);
        else
            break;
        end
    end
    while size(temp,2) > 0
        if(sum(temp(:,1)) == 0)
            temp = temp(:,2:end);
        else
            break;
        end
    end
    while size(temp,2) > 0
        if(sum(temp(:,end)) == 0)
            temp = temp(:,1:end-1);
        else
            break;
        end
    end
    
    %saving the oriented rectangle as mat file
    save(sprintf('%sconv_%d_%d_%d.mat',fileLocation,height,width,ang),'temp');
end

