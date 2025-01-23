# CounterStrike
## Setup for Starting System (Laser+LIDAR)
1. Connect the 2 USB cables.
2. Open Arduino IDE.
3. Select Board > COM5 > Nano Arduino > OK.
4. File > Examples > Firmata > StandardFirmata.
5. Tools > Processor: "..." > "..." (Old Bootloader) > Upload.
6. Run `system/main.py`.

## To Calibrate the LIDAR Motion
1. Connect the laser pointer and LIDAR to the Arduino, as described above.
2. In `system/fit.py`, uncomment the line `# measure()` in `if __name__ == '__main__':`.
3. Turn off the light, darken the room, fix the system in place.
4. Run `system/fit.py`.
5. **COPY** the output to `system/fit.py` in the `MEASUREMENTS` variable. It will automatically propagate to the rest of the system.

## Errors and Fixes
1. `"pip is not recognized"`
להריץ
`python -m pip install ...`
אם לא עובד לנסות גם:
`py -m pip install ...`
`python3 -m pip install ...`
2. If torch fails, uninstall then install again.