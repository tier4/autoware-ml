import rerun as rr
import rerun.blueprint as rrb

def setup_rerun_layout():
    """Configures the Rerun viewer to show 3D, 2D, and GICP Error plot."""
    
    # Create a blueprint with a vertical split on the right for the plot
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            # Left side: 3D View of the world
            rrb.Spatial3DView(
                name="3D LiDAR View", 
                origin="world"
            ),
            rrb.Vertical(
                # Right Top: 2D View of the camera projection
                rrb.Spatial2DView(
                    name="Virtual Camera Projection", 
                    origin="world/camera/image" 
                ),
                # Right Bottom: TimeSeries plot for GICP Error
                rrb.TimeSeriesView(
                    name="GICP Alignment Error",
                    origin="metrics"
                ),
                row_shares=[2, 1]
            ),
            column_shares=[2, 1] 
        ),
        collapse_panels=True,
    )
    
    # Send the layout to the viewer
    rr.send_blueprint(blueprint)