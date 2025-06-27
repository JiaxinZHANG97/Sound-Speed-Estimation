% Function to update the position
function updatePosition(src,num)
    % Get the current position of the rectangle
    newPosition = src.Position;

    % Store the updated position in the base workspace
    assignin('base', ['ROI_rect_',num2str(num)], newPosition);
end