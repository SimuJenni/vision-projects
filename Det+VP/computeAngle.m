function theta = computeAngle(times, frame, frontFrame, rotDir, lastFrame)
%COMPUTES THE VIEWPOINT ANGLE BASED ON THE INFORMATION PROVIDED
%   0 degrees corresponds to front frame

degPerSec = 360/times(lastFrame);
theta = rotDir*(times(frame) - times(frontFrame))*degPerSec;
if( theta > 180 ) 
    theta = theta - 360;
else if (theta < -180 )
        theta = theta + 360;
    end
end

end

