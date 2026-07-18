// Сборка C++ движка без ручного открытия "x64 Native Tools Command Prompt".
// Скрипт:
//   1. Ищет Visual Studio Build Tools через vswhere.
//   2. Готовит окружение (vcvars64.bat) для архитектуры x64.
//   3. Запускает cmake configure + build в engine/build.
//   4. Печатает путь к готовому .exe.
//
// Использование:  npm run build:engine [-- --clean]

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const ENGINE = path.join(ROOT, 'engine');
const BUILD = path.join(ENGINE, 'build');
const clean = process.argv.includes('--clean');

function log(...a)  { console.log('[build-engine]', ...a); }
function die(msg)   { console.error('[build-engine] ERROR:', msg); process.exit(1); }

// --- 1. Найти vcvars64.bat через vswhere ---
const VSWHERE = path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)',
    'Microsoft Visual Studio', 'Installer', 'vswhere.exe');

if (!fs.existsSync(VSWHERE)) {
    die(
        'Не найден vswhere.exe. Visual Studio Build Tools 2022 не установлены.\n' +
        'Установите их командой (потребуется подтверждение UAC):\n\n' +
        '  winget install Microsoft.VisualStudio.2022.BuildTools ' +
        '--override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"\n\n' +
        'После установки перезапустите терминал и повторите сборку.'
    );
}

const installationPath = spawnSync(VSWHERE,
    ['-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
     '-property', 'installationPath'],
    { encoding: 'utf8' }
).stdout.trim();

if (!installationPath) {
    die('vswhere ничего не нашёл. Установите компонент VCTools (см. README).');
}

const vcvars = path.join(installationPath, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
if (!fs.existsSync(vcvars)) die('Не найден vcvars64.bat по пути ' + vcvars);
log('Найден MSVC:', vcvars);

// --- 2. Найти cmake (обычно ставится вместе с VCTools) ---
function findCmake() {
    // Вариант 1: в PATH (cmake из VS не всегда в PATH).
    const inPath = spawnSync('where', ['cmake'], { encoding: 'utf8', shell: true }).stdout.trim();
    if (inPath) return inPath.split(/\r?\n/)[0];
    // Вариант 2: рядом с VS.
    const common = path.join(installationPath, 'Common7', 'IDE', 'CommonExtensions',
        'Microsoft', 'CMake', 'CMake', 'bin', 'cmake.exe');
    if (fs.existsSync(common)) return common;
    return null;
}

// --- 3. Очистка ---
if (clean && fs.existsSync(BUILD)) {
    log('Удаляю папку сборки (clean)...');
    fs.rmSync(BUILD, { recursive: true, force: true });
}

// --- 4. Запускаем configure + build в одной cmd-сессии с vcvars ---
//    Так переменные окружения компилятора действуют и для cmake configure, и для build.
const cmake = findCmake();
if (!cmake) {
    die('CMake не найден. Обычно он входит в состав Visual Studio VCTools.\n' +
        'Если нет — установите через winget install Kitware.CMake');
}
log('CMake:', cmake);

const vcpkgToolchain = path.join('C:\\', 'vcpkg', 'scripts', 'buildsystems', 'vcpkg.cmake');
const vcpkgArg = fs.existsSync(vcpkgToolchain) ? `-DCMAKE_TOOLCHAIN_FILE="${vcpkgToolchain}"` : '';

const cmd = [
    `"${vcvars}"`,
    `&& "${cmake}" -S "${ENGINE}" -B "${BUILD}" -A x64 ${vcpkgArg}`,
    `&& "${cmake}" --build "${BUILD}" --config Release --parallel`,
].join(' ');

const { execSync } = require('child_process');
log('Запуск сборки...');
try {
    execSync(cmd, { stdio: 'inherit' });
} catch (e) {
    die(`Сборка завершилась с ошибкой: ${e.message}`);
}
const r = { status: 0 };

if (r.status !== 0) die(`Сборка завершилась с кодом ${r.status}`);

const exeName = process.platform === 'win32' ? 'durakk_engine.exe' : 'durakk_engine';
const exeDist = path.join(ENGINE, 'dist', exeName);
const exeRoot = path.join(ROOT, exeName);
log('Готово. Собрано:');
log('  dist :', exeDist, fs.existsSync(exeDist) ? '(есть)' : '(ОТСУТСТВУЕТ)');
log('  root :', exeRoot, fs.existsSync(exeRoot) ? '(есть)' : '(ОТСУТСТВУЕТ)');
