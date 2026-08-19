; CC-Cover 安装器（Inno Setup 6）
;
; 默认当前用户安装到 {localappdata}\Programs\CC-Cover，可自选目录；
; 无需管理员权限（PrivilegesRequired=lowest）。
; 绿色压缩包为解压即用的 onedir 目录（dist\CC-Cover），数据根=exe 目录。
;
; 构建：
;   iscc.exe packaging\CC-Cover.iss
; 或从 CI 传入版本号（覆盖默认值）：
;   iscc.exe /DMyAppVersion=0.7.1 packaging\CC-Cover.iss
;
; 源目录约定：CC-Cover.iss 位于 packaging\ 下，onedir 产物位于 dist\CC-Cover\。

#ifndef MyAppVersion
  #define MyAppVersion "0.7.1"
#endif
#define MyAppName "CC-Cover"
#define MyAppExeName "CC-Cover.exe"
#define MyAppPublisher "Jinsizongzi"
#define MyAppId "{{d8dd060d-9f47-4109-b9cc-13aef3f102c7}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=CC-Cover-Setup-{#MyAppVersion}
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

; 中文语言文件随仓库自带（packaging\Languages\ChineseSimplified.isl），
; 不依赖 Inno Setup 安装是否带全语言包（choco/精简安装可能缺 ChineseSimplified.isl）。
[Languages]
Name: "chinesesimplified"; MessagesFile: "Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\CC-Cover\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

; 卸载时删除应用运行期间在数据根（=安装目录）生成的本地数据与设置。
[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\model-cache"
Type: filesandordirs; Name: "{app}\runs"
Type: filesandordirs; Name: "{app}\temp"
Type: filesandordirs; Name: "{app}\settings.json"
