%load the video
video=VideoReader("..//blink.mp4");
%get frame
video_frame=readFrame(video);

%detect the face
face_detector=vision.CascadeObjectDetector();
location_of_the_face=step(face_detector,video_frame);
%draw the face
detect_frame=insertShape(video_frame,"Rectangle",location_of_the_face);

%transform the rectangle around face 
rectangle_to_points=bbox2points(location_of_the_face(1,:));
feature_points=detectMinEigenFeatures(rgb2gray(detect_frame),"ROI",location_of_the_face);

PointTracker=vision.PointTracker('MaxBidirectionalError',2);
feature_points=feature_points.Location;
initialize(PointTracker,feature_points,detect_frame);

left=100;
bottom=100;
width=size(detect_frame,2);
height=size(detect_frame,1);
video_player=vision.VideoPlayer('Position',[left bottom width height]);

previous_points=feature_points;
while hasFrame(video)
    video_frame=readFrame(video);
    [feature_points,isFound]=step(PointTracker,video_frame);
    new_points=feature_points(isFound,:);
    old_points=previous_points(isFound,:);
    if size(new_points,1)>=2
        [transformed_rectangles,old_points,new_points]=...
            estimateGeometricTransform(old_points,new_points,'similarity','MaxDistance',4);
        rectangle_to_points=transformPointsForward(transformed_rectangles,rectangle_to_points);
        
        reshaped_rectangle=reshape(rectangle_to_points',1,[]);
        detect_frame=insertShape(video_frame,'Polygon',reshaped_rectangle,'LineWidth',2);
        detect_frame=insertMarker(detect_frame,new_points,"+",'Color','white');
        previous_points=new_points;
        setPoints(PointTracker,previous_points);
    end
    step(video_player,detect_frame);     
end
release(video_player);