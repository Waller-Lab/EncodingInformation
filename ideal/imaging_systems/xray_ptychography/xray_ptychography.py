"""
X-Ray Ptychography Imaging System Implementation.

This module implements a ptychography imaging system with both forward model
and reconstruction capabilities. The system uses a structured mask to shape
the probe beam and implements fly-scan ptychography.
"""

from ...imaging_system import ImagingSystem
import jax.numpy as np
from jax import lax, nn, random
import equinox as eqx
from typing import Optional, Tuple


import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class XRayPtychography(ImagingSystem):
    """
    X-Ray Ptychography imaging system implementation.
    
    This class implements a ptychography imaging system with both forward model
    and reconstruction capabilities. The system uses a structured mask to shape
    the probe beam.

    Attributes:
        mask: Structured illumination mask pattern
        energy: Beam energy in eV
        detector_px_sz: Detector pixel size in nm
        detector_length: Sample to detector distance in nm
        f_length: Focal length in nm
        zp_rad: Zone plate radius in nm
        bstop_rad: Beam stop radius in nm
        osa_length: Zone plate to OSA distance in nm
        osa_rad: OSA radius in nm
        window_rad: Window radius in nm
        spacing: Hole spacing in nm
        npix_detector: Number of detector pixels
        binfactor: Binning factor
        xstep: Step size in x direction
        ystep: Step size in y direction
        probe_size: Size of probe field
        n_photons: Number of incident photons
        rng_key: Random number generator key
        lambda_: X-ray wavelength in nm
        necessary_det_px_sz: Required detector pixel size
        necessary_Npx_detector: Required number of detector pixels
        px_sz_probe: Probe pixel size
        px_sz_zp: Zone plate pixel size
        px_sz_osa: OSA pixel size
        k: Wavenumber
        positions: Array of scan positions
    """
    mask: np.ndarray

    energy: int = eqx.field(static=True)
    detector_px_sz: int = eqx.field(static=True)
    detector_length: int = eqx.field(static=True)
    f_length: int = eqx.field(static=True)
    zp_rad: int = eqx.field(static=True)
    bstop_rad: int = eqx.field(static=True)
    osa_length: int = eqx.field(static=True)
    osa_rad: int = eqx.field(static=True)
    window_rad: int = eqx.field(static=True)
    spacing: int = eqx.field(static=True)
    npix_detector: int = eqx.field(static=True)
    binfactor: int = eqx.field(static=True)
    xstep: int = eqx.field(static=True)
    ystep: int = eqx.field(static=True) 
    probe_size: Tuple[int, int] = eqx.field(static=True)
    n_photons: int = eqx.field(static=True)
    rng_key: np.ndarray = eqx.field(static=True)
    lambda_: float = eqx.field(static=True)
    necessary_det_px_sz: float = eqx.field(static=True)
    necessary_Npx_detector: int = eqx.field(static=True)
    px_sz_probe: float = eqx.field(static=True)
    px_sz_zp: float = eqx.field(static=True)
    px_sz_osa: float = eqx.field(static=True)
    k: float = eqx.field(static=True)
    positions: np.ndarray = eqx.field(static=True)

    def __init__(self, energy=7e3, detector_px_sz=75e3, detector_length=2.05e9, f_length=60.5e6, zp_rad=125e3, bstop_rad=25e3, osa_length=50.5e6, osa_rad=19e3, window_rad=200, spacing=500, npix_detector=512, binfactor=1, xstep=250, ystep=250, probe_size=(512,512), n_photons=1e7, seed: int = 0):
        """
        Initialize the ptychography system.
        
        """
        self.energy = energy
        self.detector_px_sz = detector_px_sz
        self.detector_length = detector_length
        self.f_length = f_length
        self.zp_rad = zp_rad
        self.bstop_rad = bstop_rad
        self.osa_length = osa_length
        self.osa_rad = osa_rad
        self.window_rad = window_rad
        self.spacing = spacing
        self.npix_detector = npix_detector
        self.binfactor = binfactor
        self.xstep = xstep
        self.ystep = ystep
        self.probe_size = probe_size
        self.n_photons = n_photons
        self.rng_key = random.PRNGKey(seed)
        
        # Calculate derived parameters
        self._calculate_derived_params()
        
        # Initialize learnable parameters (mask)
        self._initialize_learnable_params()
        
        # Generate scan positions
        self._generate_scan_positions((800,800)) 

        super().__init__(seed)

    def _calculate_derived_params(self):
        """Calculate derived system parameters from fixed parameters."""
        # Calculate wavelength
        self.lambda_ = 1239.8 / self.energy  # nm
        
        # Calculate pixel sizes
        self.necessary_det_px_sz = self.detector_px_sz / self.binfactor
        self.necessary_Npx_detector = round(self.npix_detector * 
                                         self.detector_px_sz / 
                                         self.necessary_det_px_sz)
        
        # Calculate probe and system parameters
        self.px_sz_probe = (self.lambda_ * self.detector_length / 
                           (self.necessary_Npx_detector * self.necessary_det_px_sz))
        self.px_sz_zp = (self.necessary_det_px_sz * self.f_length / 
                        self.detector_length)
        self.px_sz_osa = (self.necessary_det_px_sz * 
                         (self.f_length - self.osa_length) / 
                         self.detector_length)
        self.k = 2 * np.pi / self.lambda_

    def _initialize_learnable_params(self):
        """Initialize learnable parameters (structured mask)."""
        # Initialize with random mask pattern
        mask_shape = (self.necessary_Npx_detector, self.necessary_Npx_detector)
        initial_mask = np.zeros(mask_shape)
        self.mask = initial_mask

    def _generate_scan_positions(self, object_shape: Tuple[int, int]):
        """
        Generate scan positions for ptychography.
        
        Args:
            object_shape: Shape of object to be scanned (height, width)
        """
        xstep, ystep = self.xstep, self.ystep
        flyrange = int(np.floor(xstep/2))
        probe_size = self.probe_size
        
        # Calculate number of scan points
        Npts_x = int(np.ceil((object_shape[1] - probe_size[1] - flyrange)/xstep)) 
        Npts_y = int(np.ceil((object_shape[0] - probe_size[0] - flyrange)/ystep))
        
        # Initialize scan positions array
        scanpos = np.zeros((Npts_x * Npts_y, 2))
        
        # Specify startpos as upper-left corner + probe.shape/2
        startpos = [np.ceil(probe_size[0]/2) + flyrange, 
                   np.ceil(probe_size[1]/2) + flyrange]
        
        # Compute scan positions
        pos_ind = 0
        for i in range(Npts_y):
            center_y = startpos[0] + i * xstep
            for j in range(Npts_x):
                center_x = startpos[1] + j * ystep
                # Add small random offset for fly scanning
                self.rng_key, key1, key2 = random.split(self.rng_key, 3)
                scanpos = scanpos.at[pos_ind].set([
                    center_y + random.randint(key1, (), -1, 2),
                    center_x + random.randint(key2, (), -1, 2)
                ])
                pos_ind += 1
        
        # Store positions in fixed params
        self.positions = scanpos

    def _generate_probe(self, mask=None) -> np.ndarray:
        """
        Generate probe field from mask pattern using Fresnel propagation.
        Matches implementation from ptycho_sim.py
        
        Returns:
            Complex probe field
        """
        if(mask is None):
            mask = nn.sigmoid(self.mask)
        
        # Get structured OSA
        _, _, structured_osa = self._gen_structured_osa(mask)
        
        # Calculate grid
        nx = int(self.necessary_Npx_detector)
        ny = int(self.necessary_Npx_detector)
        xi, eta = np.meshgrid(
            np.arange(-nx/2, nx/2),
            np.arange(-ny/2, ny/2)
        )
        
        # Generate apertures
        zp_aperture = self._gen_circle((nx, ny), self.zp_rad/self.px_sz_zp)
        bstop = 1 - self._gen_circle((nx, ny), self.bstop_rad/self.px_sz_zp)
        
        # Calculate phase term
        phase = np.exp(-(xi**2 + eta**2) * 1j * self.k * (self.px_sz_zp**2) / (2 * self.f_length))

        # First propagation (to OSA plane)
        zp_probe = zp_aperture * bstop * phase
        osa_probe = self._fresnel_back(self._fresnel_fft(zp_probe, self.px_sz_zp, self.f_length, self.lambda_), self.px_sz_probe, (self.f_length - self.osa_length), self.lambda_)
        probe = self._fresnel_fft(osa_probe * structured_osa, self.px_sz_osa, (self.f_length - self.osa_length), self.lambda_)

        # Scale probe intensity
        n_photons = self.n_photons
        probe = probe*np.sqrt(n_photons/np.sum(np.abs(probe)**2)) 
        
        return probe, zp_probe
    
    def forward_model(self, image: np.ndarray) -> np.ndarray:
        """
        Run forward model on a single image using fly-scan ptychography.
        
        Args:
            image: Input image of shape (H, W, C)
            
        Returns:
            measurements: Diffraction patterns of shape (H, W, N_positions)
        """
        probe, _ = self._generate_probe(nn.sigmoid(self.mask))
        positions = self.positions
        
        # Scale probe to desired photon count
        n_photons = self.n_photons
        probe = probe * np.sqrt(n_photons/np.sum(np.abs(probe)**2))
        
        # Initialize output array
        measurements = np.zeros((*probe.shape, len(positions)))
        
        def _scan_loop(i, measurements):
            scene = image[..., 0]  # Assume single channel
            pos = positions[i]
            flyscan_offset = 0
            lby = (pos[0] - np.floor(probe.shape[0]/2)).astype(int)
            lbx = (pos[1] - np.floor(probe.shape[1]/2) + flyscan_offset).astype(int)
            obj_curr = lax.dynamic_slice(scene, (lby, lbx), probe.shape)
            dp_single = self._my_fft(obj_curr * probe)
            measurements = measurements.at[..., i].set(np.abs(dp_single)**2)
            return measurements

        # Loop over positions
        measurements = lax.fori_loop(
            0, len(positions),
            _scan_loop,
            measurements
        )
        return measurements[..., 0]

    def _gen_structured_osa(self, structure=None):
        """
        Generate structured OSA pattern.
        
        Args:
            structure: Optional predefined structure pattern
            
        Returns:
            tuple: (structure, osa, structured_osa)
        """
        # Load Au scattering factors
        sc_au = np.load('scatteringfactors_au_highenergy.npz')['data']
        el_f1 = np.interp(self.energy, sc_au[:, 0], sc_au[:, 1])
        el_f2 = np.interp(self.energy, sc_au[:, 0], sc_au[:, 2])
        
        # Calculate physical parameters
        re = 2.818e-6  # nm
        beta = re / (2 * np.pi) * self.lambda_**2
        
        window_rad = int(self.window_rad / self.px_sz_osa)
        spacing = int(self.spacing / self.px_sz_osa)
        
        # Generate circle pattern
        circle = self._gen_circle(
            (spacing, spacing), 
            window_rad
        )
        
        if structure is None:
            structure = self._generate_binary_map(spacing, self.necessary_Npx_detector, circle)
        
        # Calculate material properties
        lacey_thickness_map = 1 - structure
        lacey_thickness_px = 1e3  # nm
        lacey_mol_weight = 197  # g/mol (Au)
        lacey_density = 19.3  # g/cm^3
        avoNum = 6.022e23
        lacey_na = lacey_density * avoNum / lacey_mol_weight / 1e21  # atoms/nm^3
        
        # Calculate complex transmission
        structure_beta = lacey_na * (lacey_thickness_px * lacey_thickness_map) * (beta*el_f2)
        structure_delta = lacey_na * (lacey_thickness_px * lacey_thickness_map) * (beta*el_f1)
        
        structure_amp = np.exp(-2*np.pi*(structure_beta)/self.lambda_)
        structure_ph = np.exp(2*1j*np.pi*(-structure_delta)/self.lambda_)
        structure = structure_amp * structure_ph
        
        # Generate OSA aperture
        osa = self._gen_circle(structure.shape, radius=self.osa_rad/self.px_sz_osa)
        structured_osa = osa * structure
        
        return structure, osa, structured_osa

    def _fresnel_fft(self, U0, res_front, z, wavelength):
        """
        Applies the Fresnel diffraction pattern to a given input field U0 using FFT.

        Parameters:
        U0 (np.ndarray): The input field.
        res_front (float): The resolution at the front plane.
        z (float): The propagation distance.
        wavelength (float): The wavelength of the light.

        Returns:
        np.ndarray: The output field after propagation.
        """
        sc = np.sum(U0)
        k = 2 * np.pi / wavelength
        ny, nx = U0.shape

        delta1 = res_front

        # Source plane coordinates
        x1, y1 = np.meshgrid(np.arange(-nx/2, nx/2) * delta1, np.arange(-ny/2, ny/2) * delta1)

        # Observation plane coordinates
        delta2 = wavelength * z / (nx * delta1)
        x2, y2 = np.meshgrid(np.arange(-nx/2, nx/2) * delta2, np.arange(-ny/2, ny/2) * delta2)

        prop_term = (np.exp(1j * k * z) / (1j / wavelength / z)) * np.exp(1j * (k / (2 * z)) * (x2**2 + y2**2))

        U = prop_term * self._my_fft(U0 * np.exp(1j * (k / (2 * z)) * (x1**2 + y1**2))) * delta1**2

        scale_to_photons = sc / np.sum(U)
        U = U * scale_to_photons

        if z == 0:
            U = U0

        return U

    def _fresnel_back(self, U0, resolution, z, wavelength):
        """
        Applies the inverse Fresnel diffraction pattern to a given input field U0 using inverse FFT.

        Parameters:
        U0 (np.ndarray): The input field.
        resolution (float): The resolution at the source plane.
        z (float): The propagation distance.
        wavelength (float): The wavelength of the light.

        Returns:
        np.ndarray: The output field after back propagation.
        """
        k = 2 * np.pi / wavelength
        ny, nx = U0.shape
        delta1 = resolution

        # Source plane coordinates
        x2, y2 = np.meshgrid(np.arange(-nx/2, nx/2) * delta1, np.arange(-ny/2, ny/2) * delta1)

        # Observation plane coordinates
        delta2 = wavelength * z / (nx * delta1)
        x1, y1 = np.meshgrid(np.arange(-nx/2, nx/2) * delta2, np.arange(-ny/2, ny/2) * delta2)

        prop_term = (np.exp(1j * k * z) / (1j / wavelength / z)) * np.exp(1j * (k / (2 * z)) * (x2**2 + y2**2))

        U = self._my_ifft(U0 / prop_term) / np.exp(1j * (k / (2 * z)) * (x1**2 + y1**2))

        return U

    def _gen_circle(self, shape: Tuple[int, int], radius: float) -> np.ndarray:
        """
        Generate binary circle mask.
        
        Args:
            shape: Output shape (height, width)
            radius: Circle radius in pixels
            
        Returns:
            Binary circle mask
        """
        height = shape[0]
        width = shape[1]

        y, x = np.meshgrid(
            np.linspace(-1, 1, height),
            np.linspace(-1, 1, width)
        )
        r = np.sqrt(x**2 + y**2)
        return (r <= radius/np.max(np.asarray(shape))).astype(np.float32)

    def _generate_binary_map(self, spacing: int, dims_int: int, circle: np.ndarray) -> np.ndarray:
        """
        Generate binary map of repeated circles.
        
        Args:
            spacing: Spacing between circles in pixels
            dims_int: Output dimensions
            circle: Single circle pattern to tile
            
        Returns:
            Binary map of tiled circles
        """
        spacing = int(spacing)
        numholes = int(np.ceil(dims_int / spacing))
        
        structure = np.tile(circle, [numholes, numholes])
        structure = structure[:dims_int, :dims_int]
        return structure.astype(np.bool_)

    def _my_fft(self, inp: np.ndarray, ax: Tuple[int, int]=(0,1), norm: str="ortho") -> np.ndarray:
        """
        Centered FFT implementation.
        
        Args:
            inp: Input array
            ax: Axes over which to compute FFT
            norm: Normalization mode
            
        Returns:
            FFT of input array
        """
        return np.fft.ifftshift(
            np.fft.fft2(
                np.fft.fftshift(inp, axes=ax), 
                axes=ax, 
                norm=norm
            ), 
            axes=ax
        )

    def _my_ifft(self, inp: np.ndarray, ax: Tuple[int, int]=(0,1), norm: str="ortho") -> np.ndarray:
        """
        Centered inverse FFT implementation.
        
        Args:
            inp: Input array
            ax: Axes over which to compute IFFT
            norm: Normalization mode
            
        Returns:
            IFFT of input array
        """
        return np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(inp, axes=ax), 
                axes=ax, 
                norm=norm
            ), 
            axes=ax
        )

    def _extract_patch(self, 
                      array: np.ndarray, 
                      position: np.ndarray, 
                      patch_size: Tuple[int, int]) -> np.ndarray:
        """
        Extract patch from array at given position.
        
        Args:
            array: Input array
            position: Center position (y, x)
            patch_size: Size of patch (height, width)
            
        Returns:
            Extracted patch
        """
        uby = int(position[0] + np.ceil(patch_size[0]/2))
        ubx = int(position[1] + np.ceil(patch_size[1]/2))
        lby = int(position[0] - np.floor(patch_size[0]/2))
        lbx = int(position[1] - np.floor(patch_size[1]/2))
        return array[lby:uby, lbx:ubx]

    def _normalize_complex(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize complex array.
        
        Args:
            arr: Complex input array
            
        Returns:
            Normalized array
        """
        return arr / (np.abs(arr) + 1e-10)

    def _pad_array(self, arr: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """
        Pad array to new shape.
        
        Args:
            arr: Input array
            new_shape: Desired shape
            
        Returns:
            Padded array
        """
        pad_y = new_shape[0] - arr.shape[0]
        pad_x = new_shape[1] - arr.shape[1]
        
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        
        return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)))

    def reconstruct(self, measurements: np.ndarray) -> np.ndarray:
        """
        Reconstruct images from measurements using ePIE algorithm.
        
        Args:
            measurements: Input measurements of shape (B, H, W, C)
            
        Returns:
            reconstructions: Reconstructed images of shape (B, H, W, C)
        """
        batch_size = measurements.shape[0]
        reconstructions = []
        
        # Process each batch independently
        for b in range(batch_size):
            measurement = measurements[b, ..., 0:1]  # Assume single channel
            
            # Run ePIE reconstruction
            obj = self._run_epie(measurement)
            reconstructions.append(obj)
        
        return np.stack(reconstructions)[..., np.newaxis]

    def _run_epie(self, 
                  measurement: np.ndarray,
                  n_iterations: int = 100,
                  alpha: float = 0.1,
                  beta: float = 0.1) -> np.ndarray:
        """
        Run ePIE reconstruction algorithm.
        
        Args:
            measurement: Diffraction patterns (H, W, N_positions)
            n_iterations: Number of iterations for reconstruction
            alpha: Object update step size
            beta: Probe update step size
            
        Returns:
            Reconstructed object
        """
        # Get parameters
        positions = self.positions
        probe, _ = self._generate_probe()
        xstep = self.xstep
        flyrange = int(np.floor(xstep/2))
        
        # Initialize object guess
        obj_shape = (
            measurement.shape[0] + int(np.max(positions[:,0]) - np.min(positions[:,0])) + flyrange + 10,
            measurement.shape[1] + int(np.max(positions[:,1]) - np.min(positions[:,1])) + flyrange + 10
        )
        obj = np.ones(obj_shape, dtype=np.complex64)
        
        # Center and normalize measurements
        measurements = np.fft.fftshift(measurement, axes=(0,1))
        
        # Main ePIE loop
        for iteration in range(n_iterations):
            # Randomly shuffle positions for this iteration
            self.rng_key, subkey = random.split(self.rng_key)
            pos_order = random.permutation(subkey, len(positions))
            
            error = 0.0
            
            for pos_idx in pos_order:
                pos = positions[pos_idx]
                
                # Extract current object patch
                obj_current = self._extract_patch(obj, pos, probe.shape)
                
                # Forward propagation
                psi = obj_current * probe
                psi_f = self._my_fft(psi)
                
                # Magnitude constraint
                amp = np.sqrt(measurements[:,:,pos_idx])
                psi_f_updated = amp * self._normalize_complex(psi_f)
                
                # Back propagation
                psi_updated = self._my_ifft(psi_f_updated)
                
                # Calculate maximum intensities for update steps
                probe_max = np.max(np.abs(probe)**2)
                obj_max = np.max(np.abs(obj_current)**2)
                
                # Update object
                obj_update = alpha * np.conj(probe) / (probe_max + 1e-10) * (psi_updated - psi)
                lby = int(pos[0]) - probe.shape[0]//2
                uby = int(pos[0]) + probe.shape[0]//2
                lbx = int(pos[1]) - probe.shape[1]//2
                ubx = int(pos[1]) + probe.shape[1]//2
                obj = obj.at[lby:uby, lbx:ubx].set(obj_current + obj_update)
                
                # Update probe
                probe_update = beta * np.conj(obj_current) / (obj_max + 1e-10) * (psi_updated - psi)
                probe = probe + probe_update
                
                # Normalize probe intensity
                probe = probe * np.sqrt(np.sum(measurements[:,:,pos_idx]) / 
                                      np.sum(np.abs(probe)**2))
                
                # Accumulate error
                error += np.mean(np.abs(np.abs(psi_f) - amp)**2)
            
            # Calculate average error for this iteration
            error = error / len(positions)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Error: {error:.3e}")
        
        return obj

    # TODO: Add display_measurement, display_object, display_optics

    def visualize_scan_positions(self, 
                               object_shape: Optional[Tuple[int, int]] = None, 
                               centers_only: bool = False,
                               save_path: Optional[str] = None):
        """
        Visualize scan positions as either circles or center points.
        
        Args:
            object_shape: Optional shape of object field (height, width)
                         If None, uses minimum bounding box of scan positions
            centers_only: If True, shows only center points; if False, shows circles
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure
        """
        # Get probe size and positions
        probe_size = self.probe_size
        positions = self.positions
        
        # Calculate object field size if not provided
        if object_shape is None:
            pad = 50  # padding around scan positions
            max_pos = np.max(positions, axis=0) + np.array(probe_size) // 2 + pad
            min_pos = np.min(positions, axis=0) - np.array(probe_size) // 2 - pad
            object_shape = tuple(max_pos - min_pos)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if centers_only:
            # Plot scan positions as points
            ax.scatter(
                positions[:, 1],  # x positions
                positions[:, 0],  # y positions
                c='blue',
                s=20,
                alpha=0.6,
                label='Scan positions'
            )
        else:
            # Plot scan positions as circles
            probe_radius = min(probe_size) // 2
            for pos in positions:
                circle = Circle(
                    (pos[1], pos[0]),  # (x, y)
                    probe_radius,
                    fill=False,
                    color='blue',
                    alpha=0.3
                )
                ax.add_patch(circle)
        
        # Set plot limits and labels
        probe_radius = min(probe_size) // 2
        ax.set_xlim(0, object_shape[1])
        ax.set_ylim(0, object_shape[0])
        ax.set_xlabel('x position (pixels)')
        ax.set_ylabel('y position (pixels)')
        ax.set_title('Scan Positions' + (' (Centers)' if centers_only else ' (Probe Areas)'))
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            suffix = '_centers' if centers_only else '_circles'
            plt.savefig(f"{save_path}_scan_positions{suffix}.png", dpi=300, bbox_inches='tight')
        
        return fig