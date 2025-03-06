@echo off
if "%debug%" == "yes" echo on

if "%1" == ""       goto syntax
if "%1" == "/?"     goto syntax
if "%1" == "unset"  goto %1

set DD_RUNNER_FAILED=
if "%DD_ROOT%"       == "" set DD_ROOT=%CD%
if "%DEVICE%"        == "" set DEVICE=stx
if "%DD_METASTATE%"  == "" if exist c:\temp\%USERNAME%\dd_metastate set DD_METASTATE=c:\temp\%USERNAME%\dd_metastate
if "%DD_METASTATE%"  == "" set DD_METASTATE=\\xsjswsvm1-abuildsFG.xilinx.com\aiebuilds\dd_ci\models_data
if "%DD_RUNNER_EXE%" == "" if exist %CD%\dd_runner.exe              set DD_RUNNER_EXE=%CD%\dd_runner.exe
if "%DD_RUNNER_EXE%" == "" set DD_RUNNER_EXE=%DD_ROOT%\build\Release\tests\dd_runner.exe

if not exist "%DD_METASTATE%"    goto bad_env
if not exist "%DD_ROOT%"         goto bad_env
if not exist "%DD_RUNNER_EXE%"   goto bad_env

goto %1

:DETAILS
pushd %DD_METASTATE%
for /f "usebackq" %%i in ( `dir /ad /b` ) do @for /f "usebackq" %%j in ( `dir /b %%i\*.state` ) do @if exist %%i\%%~nj.json for /f "usebackq" %%k in (`dir /b %%i\*.xclbin` ) do @if exist %DD_ROOT%\xclbin\%DEVICE%\%%~nk.xclbin @echo %%i  %%~nj  %%~nk.xclbin
popd
goto done

:LIST
for /f "usebackq tokens=1,2,3 delims= " %%i in ( `call %0 DETAILS` ) do @echo %%i
goto done

:DRYRUN
if     "%2" == ""  for /f "usebackq tokens=1,2,3 delims= " %%i in ( `call %0 DETAILS` ) do                        @echo %DD_RUNNER_EXE% %DD_METASTATE%\%%i\%%j.json %DD_METASTATE%\%%i\%%j.state %%k L C
if not "%2" == ""  for /f "usebackq tokens=1,2,3 delims= " %%i in ( `call %0 DETAILS` ) do if "%%i" == "%2"       @echo %DD_RUNNER_EXE% %DD_METASTATE%\%%i\%%j.json %DD_METASTATE%\%%i\%%j.state %%k L C
goto done

:RUN
for %%i in ( %0 ) do @echo %%~ni START
set DD_RUNNER_FAILED=0
if     "%2" == ""  for /f "usebackq tokens=1,2,3,4 delims= " %%i in ( `call %0 DRYRUN`    ) do echo TEST: %%i %%k %%l & %%i %%j %%k %%l L C & if ERRORLEVEL 1 set DD_RUNNER_FAILED=1
if not "%2" == ""  for /f "usebackq tokens=1,2,3,4 delims= " %%i in ( `call %0 DRYRUN %2` ) do echo TEST: %%i %%k %%l & %%i %%j %%k %%l L C & if ERRORLEVEL 1 set DD_RUNNER_FAILED=1
if %DD_RUNNER_FAILED% == 0 for %%i in ( %0 ) do @echo %%~ni COMPLETE
if %DD_RUNNER_FAILED% == 1 for %%i in ( %0 ) do @echo %%~ni COMPLETE WITH FAILURES
goto done

:RENAME
if     "%3" == ""  Error: 'rename' requires the existing name, and the new name.
if     "%3" == ""  goto syntax
if not "%3" == ""  for /f "usebackq tokens=1,2,3 delims= " %%i in ( `call %0 LIST` ) do if "%%i" == "%2"       @ren %DD_METASTATE%\%%i %3
goto done

:QUALIFY
REM Change DD_METASTATE to the vaip cache, use 'list' to find all folders that qualify, offer to run the test
set DD_TMP_METASTATE=%DD_METASTATE%
set DD_METASTATE=c:\temp\%USERNAME%\vaip\.cache
echo Searching for folders in: %DD_METASTATE%
if     "%2" == ""     call %0 LIST
if not "%2" == ""     call %0 DRYRUN %2
set DD_METASTATE=%DD_TMP_METASTATE%
if     "%2" == ""     set DD_TMP_METASTATE= & goto done

if not "%3" == "--doit" echo Press CTRL-C to skip the run... & pause
set DD_METASTATE=c:\temp\%USERNAME%\vaip\.cache
if not "%2" == ""     call %0 RUN %2
set DD_METASTATE=%DD_TMP_METASTATE%
set DD_TMP_METASTATE=
goto done

:IMPORT
REM copy a folder from the vaip .cache into DD_METASTATE
if     "%2" == ""  echo ERROR: IMPORT requires the name of a folder in the vaip cache.
if     "%2" == ""  goto done
set DD_TMP_METASTATE=%DD_METASTATE%
set DD_METASTATE=c:\temp\%USERNAME%\vaip\.cache
echo Searching for %DD_METASTATE%\%2
for /f "usebackq tokens=1,2,3 delims= " %%i in ( `call %0 DETAILS` ) do if "%%i" == "%2"  if     exist %TMP_DD_METASTATE%\%%i  echo WARNING: %%i already exists in %TMP_DD_METASTATE% - not copying.
for /f "usebackq tokens=1,2,3 delims= " %%i in ( `call %0 DETAILS` ) do if "%%i" == "%2"  if not exist %TMP_DD_METASTATE%\%%i  xcopy %DD_METASTATE%\%%i %TMP_DD_METASTATE%\%%i /khif
set DD_METASTATE=%DD_TMP_METASTATE%
set DD_TMP_METASTATE=
for /f "usebackq tokens=1,2,3 delims= " %%i in ( `call %0 DETAILS` ) do if "%%i" == "%2"  dir %DD_METASTATE%\%%i
goto done

:bad_env
echo Error: bad environment
set DD_RUNNER_FAILED=1
:SHOWENV
echo DD_METASTATE=%DD_METASTATE%
echo DD_ROOT=%DD_ROOT%
echo DEVICE=%DEVICE%
echo DD_RUNNER_EXE=%DD_RUNNER_EXE%
if     "%DD_ROOT%"       == "" echo Note: DD_ROOT        is not set.  Use the DD setup.bat.
if     "%DEVICE%"        == "" echo Note: DEVICE         is not set.  Use the DD setup.bat.
if     "%DD_METASTATE%"  == "" echo Note: DD_METASTATE   is not set.
if     "%DD_RUNNER_EXE%" == "" echo Note: DD_RUNNER_EXE  is not set.
if not "%DD_ROOT%"       == "" if not exist %DD_ROOT%       echo DD_ROOT is not valid.
if not "%DEVICE%"        == "" if not "%DEVICE%" == "stx"   echo DEVICE not recognized (stx).
if not "%DD_METASTATE%"  == "" if not exist %DD_METASTATE%  echo DD_METASTATE is not valid.  If it points to a network share, maybe try using Explorer, to log in.  (Use XLNX\accountname to log in)
if not "%DD_RUNNER_EXE%" == "" if not exist %DD_RUNNER_EXE% echo Can not find the DD_RUNNER_EXE.
goto done

:CHECKDIRS
REM Check if the env vars point to actual directories, or if files happen to exist in their place.
REM But, symbolic Links may fail with these lines (check for the NUL device that exists in every directory).
REM Default processing in this script does not use CHECKDIRS, so it does not detect existing-file or symbolic-link.
REM Use this command if the env vars look right, but things aren't working right
if not exist "%DD_METASTATE%\NUL"    goto bad_dir
if not exist "%DD_ROOT%\NUL"         goto bad_dir
goto done

:bad_dir
echo Error: bad directory name
set DD_RUNNER_FAILED=1
if not exist %DD_METASTATE%\NUL   echo DD_METASTATE=%DD_METASTATE% is not a folder.  If it points to a network share, maybe network connectivity is temporarily lost?
if not exist %DD_ROOT%\NUL        echo DD_ROOT=%DD_ROOT% is not a folder.  It should point to the DynamicDispatch repo.
echo If DD_ROOT or DD_METASTATE point to Symbolic Links, edit this script to REM out lines "if not exist" for those env vars.
echo Use "showenv" to print the env vars.
goto done

:UNSET
set DD_METASTATE=
set DD_RUNNER_EXE=
set DD_RUNNER_FAILED=
set DD_TMP_METASTATE=
goto done

:syntax
echo  Syntax: %0 command [modelfolder]
echo  Valid commands:
echo    list  details  dryrun  run
echo    showenv checkdirs unset
echo    qualify import rename
echo  This program will search DD_METASTATE for folders that dd_runner can use.
echo  In order for dd_runner to process a folder:
echo    There should be a ".state" file and a ".json" file with the same name.
echo    There should be an ".xclbin", which is also in DD_ROOT\xclbin\DEVICE.
echo  Note: these messages probably mean these things:
echo    'File Not Found' means some DD_METASTATE folders do not have all files.
echo    'The system cannot find the batch label specified' means the script
echo      did not recognize the 1st parameter as a command.
echo    'The user name or password is incorrect.' means DD_METASTATE points
echo      to a network share.  Use Explorer with that network share to log in.
echo      Use 'XLNX\username' to access the CI/CD network share.
goto done

:-h
:help
:--help
echo Commands:
echo   list      - list the folders in DD_METASTATE
echo   details   - list the folders and their state/xclbin files
echo   dryrun    - print the dd_runner.exe commands
echo   run       - launch dd_runner
echo   showenv   - print and check the env vars
echo   checkdirs - check env vars more closely, symbolic links
echo   unset     - unset env vars (not DD_ROOT or DEVICE)
echo   qualify   - run tests in the VAIP cache folder
echo   import    - copy model folder from VAIP cache folder to DD_METASTATE
echo   rename    - rename model folder in DD_METASTATE
echo Note: The script searches all folders in DD_METASTATE, and some commands will operate only on "modelfolder" when it is given.
goto done

:done
if "%DD_RUNNER_FAILED%" == "1" exit /b 1
