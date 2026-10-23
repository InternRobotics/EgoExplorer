"""Wuji glove calibration bundles.

A calibration bundle is the zip the Wuji SDK writes with
``SdkManager.export_user_data``: every hand model, tactile model and calibration run of
one SDK user.  The recorder imports it at startup (``--wuji-calibration``, default
``wuji_calibration/calibration_bundle.zip``) so that every capture machine uses the
calibration of the gloves it is connected to.  Requires ``wuji-sdk>=2026.7.21``.

Producing a bundle: see ``scripts/wuji_calibration_tool.py`` (calibrate, export).
"""

from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).resolve().parent / "calibration_bundle.zip"


def _manager(manager=None):
    if manager is not None:
        return manager
    from wuji_sdk import SdkManager
    return SdkManager.instance()


def preview_bundle(path, manager=None):
    """Validate a bundle without importing it; returns the SDK's UserDataPreview."""
    return _manager(manager).preview_user_data(str(path))


def import_and_select_bundle(path=DEFAULT_BUNDLE, manager=None):
    """Import a bundle into the local SDK and make its owner the current SDK user.

    The SDK validates the bundle first and writes nothing if validation fails.  Existing
    files of the owner user are overwritten; the owner user is created if missing.
    Returns the SDK's UserDataImportReport (owner_user_id, owner_display_name,
    device_sns, imported_at, components).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calibration bundle not found: {path}")
    manager = _manager(manager)
    report = manager.import_user_data(str(path))
    manager.switch_user(report["owner_user_id"])
    return report


def export_bundle(user_id, path=DEFAULT_BUNDLE, manager=None):
    """Export all data of one SDK user into a bundle; returns the UserDataExportReport."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _manager(manager).export_user_data(user_id, str(path))
