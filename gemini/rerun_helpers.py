import rerun as rr
import rerun.blueprint as rrb

def setup_rerun_layout():
    """Configures the Rerun viewer to show 3D and 2D side-by-side."""
    
    # Create a blueprint with a horizontal split
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            # Left side: 3D View of the world
            rrb.Spatial3DView(
                name="3D LiDAR View", 
                origin="world"
            ),
            # Right side: 2D View of the camera projection
            rrb.Spatial2DView(
                name="Virtual Camera Projection", 
                # This origin matches where we log the DepthImage or Pinhole
                origin="world/camera/image" 
            ),
            # You can adjust the split ratio (e.g., 50/50)
            column_shares=[1, 1] 
        ),
        # Automatically collapse the left/right configuration panels for a cleaner look
        collapse_panels=True,
    )
    
    # Send the layout to the viewer
    rr.send_blueprint(blueprint)