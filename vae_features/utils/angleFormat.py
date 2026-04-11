import torch


def wrap_euler_angles_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable wrap to principal range (-pi, pi] using atan2(sin, cos).
    Apply to raw decoder Euler outputs and targets before trig loss / MHR.
    """
    return torch.atan2(torch.sin(x), torch.cos(x))


def rotation_matrix_to_6d(x: torch.Tensor):
    """
    Convert Rotation matrices to 6D rotation representations
    See https://arxiv.org/pdf/1812.07035 for discussion

    Layout matches ``rotation_6d_to_matrix``: first column (3), then second
    column (3). Using ``reshape`` on the (..., 3, 2) slice would flatten
    row-major and break Gram–Schmidt recovery.

    Args:
        matrices (torch.Tensor): A tensor of shape (B, 3, 3) containing
        the rotation matrices
    """
    x = x[..., :, :2]
    return torch.cat([x[..., 0], x[..., 1]], dim=-1)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_to_rotation_matrix(euler_angles: torch.Tensor, convention: str = "XYZ"):
    """Convert 3D Euler angles (in radians) to rotation matrices.
    Args:
        euler_angles: Tensor of shape (B, 3) representing (x, y, z) angles.
    Returns:
        Tensor of shape (B, 3, 3) with rotation matrices.
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:  # noqa: PLR2004
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1), strict=False)
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def euler_to_6d(euler_angles: torch.Tensor, convention: str = "XYZ"):
    matrix = euler_to_rotation_matrix(euler_angles, convention)
    return rotation_matrix_to_6d(matrix)


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotations to 3x3 rotation matrices using Gram-Schmidt.
    Supports shape (..., 6).
    """
    if x.shape[-1] != 6:
        raise ValueError(f"Expected last dim == 6, got {x.shape[-1]}")

    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def rotation_matrix_to_euler_xyz(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to Euler XYZ angles.
    Supports shape (..., 3, 3), returns (..., 3).
    """
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrix shape (..., 3, 3), got {rotation_matrix.shape}"
        )

    sy = rotation_matrix[..., 0, 2].clamp(-1.0, 1.0)
    y = torch.asin(sy)

    x = torch.atan2(-rotation_matrix[..., 1, 2], rotation_matrix[..., 2, 2])
    z = torch.atan2(-rotation_matrix[..., 0, 1], rotation_matrix[..., 0, 0])

    # Gimbal lock handling when cos(y) ~= 0
    singular = torch.cos(y).abs() < 1e-6
    x_alt = torch.atan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 1, 1])
    z_alt = torch.zeros_like(z)

    x = torch.where(singular, x_alt, x)
    z = torch.where(singular, z_alt, z)

    return torch.stack([x, y, z], dim=-1)


def rotation_6d_to_euler(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotations to Euler XYZ angles."""
    matrix = rotation_6d_to_matrix(x)
    return rotation_matrix_to_euler_xyz(matrix)
