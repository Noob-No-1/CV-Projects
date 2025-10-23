from __future__ import annotations
import math, sys, json, argparse
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Sequence
import numpy as np

class SolveError(Exception): ...

class SolvePath:
    PROJECTION = "projection"
    FOCAL_SENSOR = "focal_sensor"
    FOCAL_PITCH = "focal_pitch"
    FOV = "fov"
    FOV_DIAG = "fov_diag"

@dataclass
class CameraSpec:
    width: Optional[int] = None
    height: Optional[int] = None
    f_mm: Optional[float] = None
    sensor_w_mm: Optional[float] = None
    sensor_h_mm: Optional[float] = None
    pitch_x_um: Optional[float] = None
    pitch_y_um: Optional[float] = None
    fov_x_deg: Optional[float] = None
    fov_y_deg: Optional[float] = None
    fov_diag_deg: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    skew: float = 0.0
    P: Optional[Sequence[Sequence[float]]] = None

    def normalized(self) -> "CameraSpec":
        w, h = self.width, self.height
        cx = self.cx if self.cx is not None else (w / 2.0 if w is not None else None)
        cy = self.cy if self.cy is not None else (h / 2.0 if h is not None else None)
        return CameraSpec(
            width=w, height=h, f_mm=self.f_mm,
            sensor_w_mm=self.sensor_w_mm, sensor_h_mm=self.sensor_h_mm,
            pitch_x_um=self.pitch_x_um, pitch_y_um=self.pitch_y_um,
            fov_x_deg=self.fov_x_deg, fov_y_deg=self.fov_y_deg, fov_diag_deg=self.fov_diag_deg,
            cx=cx, cy=cy, skew=self.skew, P=self.P
        )

class IntrinsicSolver:
    @staticmethod
    def _ensure_wh(spec: CameraSpec):
        if spec.width is None or spec.height is None:
            raise SolveError("width/height required")

    @staticmethod
    def _rq_decompose(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A_rev = np.flipud(np.fliplr(A))
        Q, R = np.linalg.qr(A_rev.T)
        R = R.T; Q = Q.T
        R = np.flipud(np.fliplr(R)); Q = np.flipud(np.fliplr(Q))
        return R, Q

    @staticmethod
    def _normalize_K(K: np.ndarray) -> np.ndarray:
        if K[2,2] == 0: raise SolveError("invalid K")
        K = K / K[2,2]
        s = np.sign(np.diag(K))
        s[s == 0] = 1
        D = np.diag(s)
        K = K @ D
        if K[0,0] <= 0 or K[1,1] <= 0:
            D2 = np.diag([1,1,1])
            if K[0,0] <= 0: D2[0,0] = -1
            if K[1,1] <= 0: D2[1,1] = -1
            K = K @ D2
        return K

    @staticmethod
    def from_projection(spec: CameraSpec) -> np.ndarray:
        if spec.P is None: raise SolveError("P missing")
        P = np.asarray(spec.P, dtype=float)
        if P.shape != (3,4): raise SolveError("P must be 3x4")
        M = P[:3,:3]
        K, _ = IntrinsicSolver._rq_decompose(M)
        return IntrinsicSolver._normalize_K(K)

    @staticmethod
    def from_focal_sensor(spec: CameraSpec) -> np.ndarray:
        IntrinsicSolver._ensure_wh(spec)
        if not all([spec.f_mm, spec.sensor_w_mm, spec.sensor_h_mm]):
            raise SolveError("focal/sensor missing")
        fx = spec.f_mm / spec.sensor_w_mm * spec.width
        fy = spec.f_mm / spec.sensor_h_mm * spec.height
        return np.array([[fx, spec.skew, spec.cx],[0., fy, spec.cy],[0.,0.,1.]])

    @staticmethod
    def from_focal_pitch(spec: CameraSpec) -> np.ndarray:
        if spec.f_mm is None or (spec.pitch_x_um is None and spec.pitch_y_um is None):
            raise SolveError("focal/pitch missing")
        px = spec.pitch_x_um if spec.pitch_x_um is not None else spec.pitch_y_um
        py = spec.pitch_y_um if spec.pitch_y_um is not None else spec.pitch_x_um
        fx = spec.f_mm * 1000.0 / px
        fy = spec.f_mm * 1000.0 / py
        return np.array([[fx, spec.skew, spec.cx],[0., fy, spec.cy],[0.,0.,1.]])

    @staticmethod
    def from_fov(spec: CameraSpec) -> np.ndarray:
        IntrinsicSolver._ensure_wh(spec)
        if spec.fov_x_deg is None and spec.fov_y_deg is None:
            raise SolveError("FOV missing")
        fx = (spec.width/2.0)/math.tan(math.radians(spec.fov_x_deg)/2.0) if spec.fov_x_deg is not None else None
        fy = (spec.height/2.0)/math.tan(math.radians(spec.fov_y_deg)/2.0) if spec.fov_y_deg is not None else None
        if fx is None and fy is not None: fx = fy * (spec.width/spec.height)
        if fy is None and fx is not None: fy = fx * (spec.height/spec.width)
        return np.array([[fx, spec.skew, spec.cx],[0., fy, spec.cy],[0.,0.,1.]])

    @staticmethod
    def from_fov_diag(spec: CameraSpec) -> np.ndarray:
        IntrinsicSolver._ensure_wh(spec)
        if spec.fov_diag_deg is None: raise SolveError("diag FOV missing")
        half_diag = 0.5 * math.sqrt(spec.width**2 + spec.height**2)
        fd = half_diag / math.tan(math.radians(spec.fov_diag_deg)/2.0)
        return np.array([[fd, spec.skew, spec.cx],[0., fd, spec.cy],[0.,0.,1.]])

    @staticmethod
    def solve(spec: CameraSpec) -> Tuple[np.ndarray, str]:
        s = spec.normalized()
        try:
            if s.P is not None: return IntrinsicSolver.from_projection(s), SolvePath.PROJECTION
        except SolveError: pass
        try:
            if s.f_mm and s.sensor_w_mm and s.sensor_h_mm and s.width and s.height:
                return IntrinsicSolver.from_focal_sensor(s), SolvePath.FOCAL_SENSOR
        except SolveError: pass
        try:
            if s.f_mm and (s.pitch_x_um or s.pitch_y_um):
                return IntrinsicSolver.from_focal_pitch(s), SolvePath.FOCAL_PITCH
        except SolveError: pass
        try:
            if (s.fov_x_deg is not None or s.fov_y_deg is not None) and s.width and s.height:
                return IntrinsicSolver.from_fov(s), SolvePath.FOV
        except SolveError: pass
        try:
            if s.fov_diag_deg is not None and s.width and s.height:
                return IntrinsicSolver.from_fov_diag(s), SolvePath.FOV_DIAG
        except SolveError: pass
        raise SolveError("insufficient information")

def calibrate_from_chessboard(image_paths: Iterable[str], pattern_size: Tuple[int,int]=(9,6), square_size: float=1.0):
    try:
        import cv2 as cv
    except Exception as e:
        raise RuntimeError("opencv-python required") from e
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
    objp *= float(square_size)
    objpoints, imgpoints, imsize = [], [], None
    for p in image_paths:
        img = cv.imread(p, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])
        ok, corners = cv.findChessboardCorners(img, pattern_size)
        if ok:
            corners2 = cv.cornerSubPix(img, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            objpoints.append(objp)
            imgpoints.append(corners2)
    if not objpoints:
        raise RuntimeError("no chessboard detected")
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imsize, None, None)
    return K, dist, ret

def _demo_cases():
    cases = []
    P = None
    fx, fy, cx, cy = 1200., 1180., 960., 540.
    Ktrue = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    R = np.eye(3)
    t = np.array([[0],[0],[-1000]])
    P = Ktrue @ np.hstack([R, t])
    cases.append(CameraSpec(P=P))
    cases.append(CameraSpec(width=6000, height=4000, f_mm=35., sensor_w_mm=36., sensor_h_mm=24.))
    cases.append(CameraSpec(width=4000, height=3000, f_mm=4.25, pitch_x_um=1.55, pitch_y_um=1.55))
    cases.append(CameraSpec(width=1920, height=1080, fov_x_deg=60.))
    cases.append(CameraSpec(width=3840, height=2160, fov_diag_deg=75.))
    return cases

def _np_to_list(M: np.ndarray):
    return [[float(M[i,j]) for j in range(M.shape[1])] for i in range(M.shape[0])]

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--spec", type=str, default="", help="json string or path")
    ap.add_argument("--chessboard", type=str, nargs="*", default=None)
    ap.add_argument("--pattern", type=str, default="9x6")
    ap.add_argument("--square", type=float, default=1.0)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.chessboard:
        a,b = args.pattern.lower().split("x")
        a,b = int(a), int(b)
        K, dist, err = calibrate_from_chessboard(args.chessboard, (a,b), args.square)
        print(json.dumps({"K": _np_to_list(K), "dist": [float(x) for x in dist.ravel()], "reproj_error": float(err)}, indent=2))
        return

    if args.demo:
        out = []
        for i, spec in enumerate(_demo_cases()):
            try:
                K, path = IntrinsicSolver.solve(spec)
                out.append({"case": i+1, "path": path, "K": _np_to_list(K)})
            except Exception as e:
                out.append({"case": i+1, "error": str(e)})
        print(json.dumps(out, indent=2))
        return

    if args.spec:
        try:
            if args.spec.endswith(".json"):
                with open(args.spec, "r", encoding="utf-8") as f:
                    d = json.load(f)
            else:
                d = json.loads(args.spec)
        except Exception:
            print("invalid spec", file=sys.stderr); sys.exit(2)
        spec = CameraSpec(
            width=d.get("width"), height=d.get("height"),
            f_mm=d.get("f_mm"), sensor_w_mm=d.get("sensor_w_mm"), sensor_h_mm=d.get("sensor_h_mm"),
            pitch_x_um=d.get("pitch_x_um"), pitch_y_um=d.get("pitch_y_um"),
            fov_x_deg=d.get("fov_x_deg"), fov_y_deg=d.get("fov_y_deg"), fov_diag_deg=d.get("fov_diag_deg"),
            cx=d.get("cx"), cy=d.get("cy"), skew=d.get("skew", 0.0),
            P=d.get("P")
        )
        try:
            K, path = IntrinsicSolver.solve(spec)
            print(json.dumps({"path": path, "K": _np_to_list(K)}, indent=2))
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr); sys.exit(1)
        return

    print("usage examples:")
    print("  python intrinsics.py --demo")
    print('  python intrinsics.py --spec "{\"width\":6000,\"height\":4000,\"f_mm\":35,\"sensor_w_mm\":36,\"sensor_h_mm\":24}"')
    print('  python intrinsics.py --spec spec.json')
    print("  python intrinsics.py --chessboard imgs/*.png --pattern 9x6 --square 1.0")

if __name__ == "__main__":
    main()
