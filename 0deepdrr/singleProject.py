import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector


def main():
    output_dir = test_utils.get_output_dir()
    patient = deepdrr.Volume.from_nifti(
    "dataset2_1.3.6.1.4.1.9328.50.4.0370.nii.gz", use_thresholding=True, 
    )
    patient.faceup()

    # define the simulated C-arm
    carm = deepdrr.MobileCArm(patient.center_in_world + geo.v(50, 0, -300))

    # project in the AP view
    with Projector(patient, carm=carm) as projector:
        carm.move_to(alpha=0, beta=10)
        image = projector()

    path = output_dir / "example_projector.png"
    image_utils.save(path, image)
    print(f"saved example projection image to {path.absolute()}")


if __name__ == "__main__":
    main()