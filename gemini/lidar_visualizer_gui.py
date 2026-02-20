import os
import json
import pickle
import numpy as np
import rerun as rr

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from rerun_helpers import setup_rerun_layout

class LidarFusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LiDAR-LiDAR Miscalibration Pipeline")
        self.root.geometry("600x900")

        # Safer session storage in user's home directory
        self.session_file = os.path.expanduser("~/.lidar_fusion_session.json")

        # Application State
        self.dataset_root = ""
        self.current_pkl_path = ""
        self.pickle_data = []
        self.selected_frame_data = None
        
        # 3D Point State (Ego Frame)
        self.lidar1_raw_pts = None
        self.lidar2_raw_pts = None
        self.lidar1_pts = None
        self.lidar2_pts = None
        self.combined_pts = None
        self.lidar1_ext = None
        self.lidar2_ext = None
        self.current_frame_id = ""
        
        # Default Parameters
        self.extrinsics = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'Roll': 0.0, 'Pitch': 0.0, 'Yaw': 0.0}
        self.intrinsics = {'fx': 600.0, 'fy': 600.0, 'cx': 640.0, 'cy': 360.0, 'width': 1280.0, 'height': 720.0}
        self.noise_params = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'Roll': 0.0, 'Pitch': 0.0, 'Yaw': 0.0}

        # Tkinter Variables
        self.ext_vars = {}
        self.int_vars = {}
        self.noise_vars = {}
        self.method_var = tk.StringVar(value="global")
        self.apply_transforms_var = tk.BooleanVar(value=False)
        
        # Debouncing state
        self.debounce_id = None

        # Build Initial UI
        self.build_setup_ui()
        
        # Load last session if it exists
        self.load_session()

    # ==========================================
    # PIPELINE CONTROL (DEBOUNCING & SPLITTING)
    # ==========================================
    def trigger_3d_update(self, *args):
        """Triggered by Noise sliders and Transform toggle. Re-calculates 3D points."""
        if self.debounce_id:
            self.root.after_cancel(self.debounce_id)
        self.debounce_id = self.root.after(50, self.update_3d_scene)

    def trigger_camera_update(self, *args):
        """Triggered by Camera/Projection sliders. Uses existing transformed 3D points."""
        if self.debounce_id:
            self.root.after_cancel(self.debounce_id)
        self.debounce_id = self.root.after(50, self.update_camera_and_projections)

    def update_3d_scene(self):
        """Applies 3D transforms to LiDAR points and logs them. Cascades to camera update."""
        self.debounce_id = None
        if self.lidar1_raw_pts is None or self.lidar2_raw_pts is None:
            return

        # 1. Base Alignment (Optional)
        apply_tx = self.apply_transforms_var.get()
        if apply_tx:
            self.lidar1_pts = self.apply_transformation(self.lidar1_raw_pts, self.lidar1_ext['R'], self.lidar1_ext['T'])
            self.lidar2_pts = self.apply_transformation(self.lidar2_raw_pts, self.lidar2_ext['R'], self.lidar2_ext['T'])
        else:
            self.lidar1_pts = self.lidar1_raw_pts.copy()
            self.lidar2_pts = self.lidar2_raw_pts.copy()
        
        # 2. Apply Noise to LiDAR 2 (Applied in Ego Frame)
        noise_vals = [self.noise_vars[d].get() for d in ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']]
        M_noise = self.get_4x4_matrix(noise_vals[:3], noise_vals[3:])
        self.lidar2_pts = self.apply_transformation(self.lidar2_pts, M_noise[:3, :3], M_noise[:3, 3])
        
        self.combined_pts = np.vstack([self.lidar1_pts, self.lidar2_pts])

        # 3. Log 3D to Rerun
        rr.log("world/lidar1", rr.Points3D(self.lidar1_pts, colors=[255, 50, 50]), static=True)
        rr.log("world/lidar2", rr.Points3D(self.lidar2_pts, colors=[50, 150, 255]), static=True)

        # 4. Cascade to camera and projection logic
        self.update_camera_and_projections()

    def update_camera_and_projections(self):
        """Calculates camera pose and projects existing 3D points to 2D image frames."""
        self.debounce_id = None
        if self.combined_pts is None:
            return

        # 1. Capture parameters from state
        t_wc = [self.ext_vars[d].get() for d in ['X', 'Y', 'Z']]
        r_wc_deg = [self.ext_vars[d].get() for d in ['Roll', 'Pitch', 'Yaw']]
        
        fx = self.int_vars['fx'].get()
        fy = self.int_vars['fy'].get()
        cx = self.int_vars['cx'].get()
        cy = self.int_vars['cy'].get()
        w = int(self.int_vars['width'].get())
        h = int(self.int_vars['height'].get())
        
        method = self.method_var.get()

        # 2. Compute Camera Rotation/Translation
        R_wc, t_wc_vec = self.get_camera_matrices(t_wc, r_wc_deg)

        # 3. Log Camera to Rerun
        rr.log("world/camera", rr.Transform3D(translation=t_wc_vec, mat3x3=R_wc), static=True)
        rr.log("world/camera/image", rr.Pinhole(
            image_from_camera=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            resolution=[w, h], camera_xyz=rr.ViewCoordinates.RDF
        ), static=True)

        # 4. Clear Projections
        rr.log("world/camera/image/projections", rr.Clear(recursive=True), static=True)
        rr.log("world/camera/image/depth_map", rr.Clear(recursive=False), static=True)
        rr.log("world/camera/image/spherical", rr.Clear(recursive=False), static=True)
        rr.log("world/camera/image/global", rr.Clear(recursive=True), static=True)

        # 5. Project points to Camera Frame (RDF)
        pts_c = (self.combined_pts - t_wc_vec) @ R_wc

        # 6. Execute Projections
        if method == "depth_map":
            depth_img = self.project_depth_map(pts_c, fx, fy, cx, cy, w, h)
            rr.log("world/camera/image/projections/depth_map", rr.DepthImage(depth_img, meter=1.0), static=True)
        elif method == "spherical":
            range_img = self.project_spherical(pts_c, w, h)
            rr.log("world/camera/image/projections/spherical", rr.DepthImage(range_img, meter=1.0), static=True)
        elif method == "global":
            # Project points to 2D pixels for the pinhole view
            z = pts_c[:, 2]
            valid = z > 0.1
            if np.any(valid):
                pts_c_v = pts_c[valid]
                u = (fx * pts_c_v[:, 0] / pts_c_v[:, 2]) + cx
                v = (fy * pts_c_v[:, 1] / pts_c_v[:, 2]) + cy
                
                valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if np.any(valid_uv):
                    u, v = u[valid_uv], v[valid_uv]
                    
                    # Distinguish lidar1 and lidar2 for coloring
                    n1 = len(self.lidar1_pts)
                    orig_indices = np.where(valid)[0][valid_uv]
                    is_lidar1 = orig_indices < n1
                    
                    points_2d = np.column_stack([u, v])
                    colors = np.zeros((len(points_2d), 3), dtype=np.uint8)
                    colors[is_lidar1] = [255, 50, 50]
                    colors[~is_lidar1] = [50, 150, 255]
                    
                    rr.log("world/camera/image/projections/perspective", rr.Points2D(points_2d, colors=colors), static=True)

    # ==========================================
    # DECOUPLED MATH LOGIC
    # ==========================================
    def get_camera_matrices(self, t, r_deg):
        M = self.get_4x4_matrix(t, r_deg)
        return M[:3, :3], M[:3, 3]

    def project_depth_map(self, pts_c, fx, fy, cx, cy, w, h):
        """Projects points to a Z-buffer depth map."""
        z = pts_c[:, 2]
        valid = z > 0.1
        pts_c, z = pts_c[valid], z[valid]

        u = np.round((fx * pts_c[:, 0] / z) + cx).astype(int)
        v = np.round((fy * pts_c[:, 1] / z) + cy).astype(int)

        valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[valid_uv], v[valid_uv], z[valid_uv]

        depth_img = np.zeros((h, w), dtype=np.float32)
        sort_idx = np.argsort(z)[::-1] # Ensure closest points are drawn last
        depth_img[v[sort_idx], u[sort_idx]] = z[sort_idx]
        return depth_img

    def project_spherical(self, pts_c, w, h):
        """Projects points to a range image using Azimuth/Elevation mapping (RDF)."""
        x, y, z_c = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        r = np.linalg.norm(pts_c, axis=1)
        r_clip = np.clip(r, 1e-5, None)
        
        # RDF Frame: x=Right, y=Down, z=Forward
        azimuth = np.arctan2(x, z_c) 
        elevation = np.arcsin(-y / r_clip) # -y maps Up to positive values

        # Map angles to pixel coordinates
        u = np.round((azimuth / np.pi + 1.0) * 0.5 * (w - 1)).astype(int)
        v = np.round((1.0 - (elevation / (np.pi / 2))) * 0.5 * (h - 1)).astype(int)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, r = u[valid], v[valid], r[valid]

        range_img = np.zeros((h, w), dtype=np.float32)
        range_img[v, u] = r
        return range_img

    def get_4x4_matrix(self, t, r_deg):
        """Creates a 4x4 transformation matrix from translation and ZYX Euler rotations."""
        r, p, y = map(np.radians, r_deg)
        c_y, s_y = np.cos(y), np.sin(y)
        c_p, s_p = np.cos(p), np.sin(p)
        c_r, s_r = np.cos(r), np.sin(r)

        R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
        R_y = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
        R_x = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
        R = R_z @ R_y @ R_x
        
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def apply_transformation(self, points, R, T):
        """Applies rotation and translation to Nx3 points."""
        return points @ R.T + T

    # ==========================================
    # STEP 1: READ INPUTS & SETUP UI
    # ==========================================
    def build_setup_ui(self):
        """Builds the initial screen to select data files and LiDARs."""
        self.setup_frame = ttk.Frame(self.root, padding=20)
        self.setup_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.setup_frame, text="1. Select Dataset Base Folder (Optional)").pack(anchor="w")
        self.btn_base = ttk.Button(self.setup_frame, text="Select Base Dir", command=self.load_base_dir)
        self.btn_base.pack(fill=tk.X, pady=5)

        ttk.Label(self.setup_frame, text="2. Select Pickle File").pack(anchor="w", pady=(10, 0))
        self.btn_pkl = ttk.Button(self.setup_frame, text="Load Pickle", command=self.load_pickle)
        self.btn_pkl.pack(fill=tk.X, pady=5)

        ttk.Label(self.setup_frame, text="3. Select Frame").pack(anchor="w", pady=(10, 0))
        self.frame_combo = ttk.Combobox(self.setup_frame, state="readonly")
        self.frame_combo.bind("<<ComboboxSelected>>", self.on_frame_selected)
        self.frame_combo.pack(fill=tk.X, pady=5)

        ttk.Label(self.setup_frame, text="4. Select Exactly 2 LiDAR Sources").pack(anchor="w", pady=(10, 0))
        self.lidar_listbox = tk.Listbox(self.setup_frame, selectmode=tk.MULTIPLE, height=8)
        self.lidar_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        self.btn_launch = ttk.Button(self.setup_frame, text="Process Data & Launch Rerun", command=self.process_and_launch)
        self.btn_launch.pack(fill=tk.X, pady=20)

    def load_base_dir(self):
        self.dataset_root = filedialog.askdirectory(title="Select Dataset Base Directory")
        if self.dataset_root:
            self.btn_base.config(text=f"Base: {self.dataset_root}")

    def load_pickle(self, pkl_path=None):
        if not pkl_path:
            pkl_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if not pkl_path: return
        
        if not self.dataset_root:
            self.dataset_root = os.path.dirname(pkl_path)
            self.btn_base.config(text=f"Base: {os.path.basename(self.dataset_root)}")

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            self.pickle_data = data.get('data_list', [])

        if not self.pickle_data:
            messagebox.showerror("Error", "No 'data_list' found in pickle.")
            return

        self.current_pkl_path = pkl_path
        self.save_session(pkl_path)

        frame_names = [f"{i}: {d.get('frame_id')} (Idx: {d.get('frame_idx')})" for i, d in enumerate(self.pickle_data)]
        self.frame_combo['values'] = frame_names
        self.frame_combo.current(0)
        self.on_frame_selected(None)
        self.btn_pkl.config(text=f"Loaded: {os.path.basename(pkl_path)}")

    def on_frame_selected(self, event):
        idx = self.frame_combo.current()
        self.selected_frame_data = self.pickle_data[idx]
        sources = self.selected_frame_data.get('lidar_sources', {}).keys()
        
        self.lidar_listbox.delete(0, tk.END)
        for src in sources:
            if src != "LIDAR_CONCAT":
                self.lidar_listbox.insert(tk.END, src)

    def process_and_launch(self):
        selections = self.lidar_listbox.curselection()
        if len(selections) != 2:
            messagebox.showerror("Error", "Please select exactly 2 LiDAR sources.")
            return

        src1 = self.lidar_listbox.get(selections[0])
        src2 = self.lidar_listbox.get(selections[1])

        try:
            self.extract_and_transform_points(src1, src2)
            self.setup_frame.destroy()
            rr.init("LiDAR_Miscalibration", spawn=True)
            setup_rerun_layout()
            self.build_control_ui()
            self.update_3d_scene() # Initial full update
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def extract_and_transform_points(self, src1, src2):
        """Extracts points from BIN using JSON indices and stores them with their extrinsics."""
        lidar_path = self.selected_frame_data['lidar_points']['lidar_path']
        abs_bin_path = os.path.join(self.dataset_root, lidar_path)
        abs_json_path = abs_bin_path.replace("LIDAR_CONCAT", "LIDAR_CONCAT_INFO").replace(".pcd.bin", ".json")

        if not os.path.exists(abs_bin_path) or not os.path.exists(abs_json_path):
            raise FileNotFoundError(f"Missing BIN or JSON file.\nLooked for:\n{abs_bin_path}\n{abs_json_path}")

        sources_dict = self.selected_frame_data['lidar_sources']
        self.lidar1_ext = {'R': np.array(sources_dict[src1]['rotation']), 'T': np.array(sources_dict[src1]['translation'])}
        self.lidar2_ext = {'R': np.array(sources_dict[src2]['rotation']), 'T': np.array(sources_dict[src2]['translation'])}

        with open(abs_json_path, 'r') as f:
            info = json.load(f)

        token1, token2 = sources_dict[src1]['sensor_token'], sources_dict[src2]['sensor_token']
        idx_len_1 = idx_len_2 = None
        for s in info['sources']:
            if s['sensor_token'] == token1: idx_len_1 = (s['idx_begin'], s['length'])
            if s['sensor_token'] == token2: idx_len_2 = (s['idx_begin'], s['length'])

        raw_data = np.fromfile(abs_bin_path, dtype=np.float32)
        total_pts = sum(s['length'] for s in info['sources'])
        num_features = len(raw_data) // total_pts
        points_nx3 = raw_data.reshape(-1, num_features)[:, :3]

        self.lidar1_raw_pts = points_nx3[idx_len_1[0] : idx_len_1[0] + idx_len_1[1]]
        self.lidar2_raw_pts = points_nx3[idx_len_2[0] : idx_len_2[0] + idx_len_2[1]]
        self.current_frame_id = f"{lidar_path}_{src1}_{src2}"

    # ==========================================
    # STEP 2: TKINTER CONTROL PANEL (MAIN UI)
    # ==========================================
    def build_control_ui(self):
        """Builds the parameter control panel after data is loaded."""
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Extrinsics (Trigger Camera Update)
        ext_lf = ttk.LabelFrame(self.control_frame, text="Virtual Camera Extrinsics (Ego Frame)", padding=10)
        ext_lf.pack(fill=tk.X, pady=5)
        for i, dim in enumerate(['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']):
            ttk.Label(ext_lf, text=dim).grid(row=i//2, column=(i%2)*2, sticky="e", padx=5)
            var = tk.DoubleVar(value=self.extrinsics[dim])
            v_min, v_max = (-20, 20) if dim in ['X','Y','Z'] else (-180, 180)
            inc = 0.1 if dim in ['X','Y','Z'] else 1.0
            spin = ttk.Spinbox(ext_lf, from_=v_min, to=v_max, increment=inc, textvariable=var, command=self.trigger_camera_update)
            spin.grid(row=i//2, column=(i%2)*2+1, sticky="we", padx=5)
            spin.bind("<Return>", lambda e: self.trigger_camera_update())
            self.ext_vars[dim] = var

        # 2. Noise Transform (Trigger 3D Update)
        noise_lf = ttk.LabelFrame(self.control_frame, text="LiDAR 2 Noise Transform (Ego Frame)", padding=10)
        noise_lf.pack(fill=tk.X, pady=5)
        for i, dim in enumerate(['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']):
            ttk.Label(noise_lf, text=dim).grid(row=i//2, column=(i%2)*2, sticky="e", padx=5)
            var = tk.DoubleVar(value=self.noise_params[dim])
            v_min, v_max = (-2, 2) if dim in ['X','Y','Z'] else (-10, 10)
            inc = 0.01 if dim in ['X','Y','Z'] else 0.1
            spin = ttk.Spinbox(noise_lf, from_=v_min, to=v_max, increment=inc, textvariable=var, command=self.trigger_3d_update)
            spin.grid(row=i//2, column=(i%2)*2+1, sticky="we", padx=5)
            spin.bind("<Return>", lambda e: self.trigger_3d_update())
            self.noise_vars[dim] = var

        # 3. Intrinsics (Trigger Camera Update)
        int_lf = ttk.LabelFrame(self.control_frame, text="Virtual Camera Intrinsics", padding=10)
        int_lf.pack(fill=tk.X, pady=5)
        for i, dim in enumerate(['fx', 'fy', 'cx', 'cy', 'width', 'height']):
            ttk.Label(int_lf, text=dim).grid(row=i//2, column=(i%2)*2, sticky="e", padx=5)
            var = tk.DoubleVar(value=self.intrinsics[dim])
            spin = ttk.Spinbox(int_lf, from_=1, to=4000, increment=10, textvariable=var, command=self.trigger_camera_update)
            spin.grid(row=i//2, column=(i%2)*2+1, sticky="we", padx=5)
            spin.bind("<Return>", lambda e: self.trigger_camera_update())
            self.int_vars[dim] = var

        # 4. Options (Trigger 3D or Camera Update)
        options_lf = ttk.LabelFrame(self.control_frame, text="Options & Projection Method", padding=10)
        options_lf.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(options_lf, text="Apply LiDAR-to-Ego Transforms", variable=self.apply_transforms_var, command=self.trigger_3d_update).pack(anchor="w", pady=(0, 5))
        ttk.Separator(options_lf, orient="horizontal").pack(fill=tk.X, pady=5)
        ttk.Radiobutton(options_lf, text="Unified Global Frame", variable=self.method_var, value="global", command=self.trigger_camera_update).pack(anchor="w")
        ttk.Radiobutton(options_lf, text="Depth Map Accumulation", variable=self.method_var, value="depth_map", command=self.trigger_camera_update).pack(anchor="w")
        ttk.Radiobutton(options_lf, text="Spherical Panoramic", variable=self.method_var, value="spherical", command=self.trigger_camera_update).pack(anchor="w")

        # 5. Actions
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        ttk.Button(btn_frame, text="Front View", command=self.set_front_view).pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(btn_frame, text="Top View", command=self.set_top_view).pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(btn_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, expand=True, padx=5)

    def reset_defaults(self):
        """Resets all variables to default and triggers a full scene update."""
        for dim, val in self.extrinsics.items(): self.ext_vars[dim].set(val)
        for dim, val in self.noise_params.items(): self.noise_vars[dim].set(val)
        for dim, val in self.intrinsics.items(): self.int_vars[dim].set(val)
        self.method_var.set("global")
        self.apply_transforms_var.set(False)
        self.trigger_3d_update()

    # ==========================================
    # SESSION & CONFIG
    # ==========================================
    def save_session(self, pkl_path=None):
        session = {"dataset_root": self.dataset_root, "last_pickle": pkl_path or self.current_pkl_path}
        try:
            with open(self.session_file, 'w') as f: json.dump(session, f)
        except Exception: pass

    def load_session(self):
        if not os.path.exists(self.session_file): return
        try:
            with open(self.session_file, 'r') as f: session = json.load(f)
            self.dataset_root = session.get("dataset_root", "")
            if self.dataset_root and os.path.exists(self.dataset_root):
                self.btn_base.config(text=f"Base: {os.path.basename(self.dataset_root)}")
            pkl_path = session.get("last_pickle", "")
            if pkl_path and os.path.exists(pkl_path): self.load_pickle(pkl_path)
        except Exception: pass

    def save_config(self):
        config = {
            "method": self.method_var.get(),
            "apply_transforms": self.apply_transforms_var.get(),
            "extrinsics": {k: v.get() for k, v in self.ext_vars.items()},
            "noise": {k: v.get() for k, v in self.noise_vars.items()},
            "intrinsics": {k: v.get() for k, v in self.int_vars.items()}
        }
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'w') as f: json.dump(config, f, indent=4)
            messagebox.showinfo("Saved", "Configuration saved successfully!")

    def set_front_view(self):
        """Places camera between lidars looking at the normal of the connecting line."""
        if self.lidar1_ext is None or self.lidar2_ext is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        p1 = self.lidar1_ext['T']
        p2 = self.lidar2_ext['T']
        
        # Midpoint position
        t_cam = (p1 + p2) / 2.0
        
        # Vector from lidar1 to lidar2
        v = p2 - p1
        
        # Normal to the line in XY plane: (dx, dy) -> (-dy, dx)
        # Angle of the normal vector
        yaw = np.arctan2(v[0], -v[1]) 
        
        self.ext_vars['X'].set(round(t_cam[0], 2))
        self.ext_vars['Y'].set(round(t_cam[1], 2))
        self.ext_vars['Z'].set(round(t_cam[2], 2))
        self.ext_vars['Roll'].set(-90.0)
        self.ext_vars['Pitch'].set(0.0)
        self.ext_vars['Yaw'].set(round(np.degrees(yaw), 2))
        self.trigger_camera_update()

    def set_top_view(self):
        """Bird's eye view covering lateral [-10, 10] and longitudinal [-10, 100]."""
        # Center of x [-10, 10] and y [-10, 100] in nuScenes Ego
        # Center: x=45 (longitudinal), y=0 (lateral). Height=100.
        # Orientation: Looking Down, Forward is Up in image.
        self.ext_vars['X'].set(45.0)
        self.ext_vars['Y'].set(0.0)
        self.ext_vars['Z'].set(100.0)
        self.ext_vars['Roll'].set(180.0)
        self.ext_vars['Pitch'].set(0.0)
        self.ext_vars['Yaw'].set(-90.0)
        
        w = self.int_vars['width'].get()
        h = self.int_vars['height'].get()
        # fx fits 20m lateral (nuScenes Y) -> width w
        fx = (w / 2.0) / (10.0 / 100.0)
        # fy fits 110m longitudinal (nuScenes X) -> height h
        fy = (h / 2.0) / (55.0 / 100.0)
        
        f = min(fx, fy)
        self.int_vars['fx'].set(round(f, 1))
        self.int_vars['fy'].set(round(f, 1))
        self.int_vars['cx'].set(round(w / 2.0, 1))
        self.int_vars['cy'].set(round(h / 2.0, 1))
        self.trigger_camera_update()

if __name__ == "__main__":
    root = tk.Tk()
    app = LidarFusionApp(root)
    root.mainloop()
