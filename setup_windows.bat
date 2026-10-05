@echo off
echo === GymPose-Lite Windows Setup ===
echo.
echo Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo Python not found! Install from https://www.python.org/downloads/
    echo Make sure to check "Add to PATH" during install.
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pyyaml matplotlib tqdm numpy pycocotools

echo.
echo Setup complete! Run the demo with:
echo   python demo.py --checkpoint checkpoints/distill_test.pth --use-cv
echo.
echo Controls: 1=Squat, 2=Push-up, 3=Deadlift, q=Quit
pause
