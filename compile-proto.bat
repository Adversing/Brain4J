@echo off
set SCRIPT_DIR=%~dp0

set PROTO_PATH=%SCRIPT_DIR%brain4j-core\src\main\resources
set PROTO_FILE=%PROTO_PATH%\model.proto
set OUT_DIR=%SCRIPT_DIR%brain4j-core\src\main\java

if not exist "%PROTO_FILE%" (
  echo ERRORE: File %PROTO_FILE% was not found
  exit /b 1
)

if not exist "%OUT_DIR%" (
  mkdir "%OUT_DIR%"
  if errorlevel 1 (
    echo ERROR: Failed to create directory %OUT_DIR%
    exit /b 1
  )
)

protoc ^
  --proto_path="%PROTO_PATH%" ^
  --java_out="%OUT_DIR%" ^
  "%PROTO_FILE%"

if errorlevel 1 (
  echo ERRORE: protoc reported an error during generation
  exit /b 1
)

echo Generated protobuf successfully in %OUT_DIR%
exit /b 0
