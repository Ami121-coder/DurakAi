// Сборка C++ движка без ручного открытия консоли компилятора.
//
// FIX #21: ранее скрипт был полностью виндовым (vswhere.exe, vcvars64.bat).
// Теперь поддерживает Windows, Linux, macOS:
//   • Windows: ищет VS Build Tools через vswhere, поднимает vcvars64.bat.
//   • Linux/macOS: проверяет наличие gcc/clang и cmake в PATH.
//
// Использование:  npm run build:engine [-- --clean]

const fs = require('fs');
const path = require('path');
const { spawnSync, execSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const ENGINE = path.join(ROOT, 'engine');
const BUILD = path.join(ENGINE, 'build');
const clean = process.argv.includes('--clean');

function log(...a)  { console.log('[build-engine]', ...a); }
function die(msg)   { console.error('[build-engine] ERROR:', msg); process.exit(1); }

const isWin = process.platform === 'win32';
const isMac = process.platform === 'darwin';

// --- Поиск cmake в PATH ---
function findCmake() {
    const cmd = isWin ? 'where cmake' : 'which cmake';
    try {
        const out = spawnSync(cmd, [], { encoding: 'utf8', shell: true }).stdout.trim();
        if (out) return out.split(/\r?\n/)[0];
    } catch (_) {}
    return null;
}

// --- Очистка ---
if (clean && fs.existsSync(BUILD)) {
    log('Удаляю папку сборки (clean)...');
    fs.rmSync(BUILD, { recursive: true, force: true });
}

// ===========================================================================
// Windows: поднимаем vcvars64.bat через vswhere
// ===========================================================================
let envSetupCmd = '';  // команда для активации окружения компилятора

if (isWin) {
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

    // Поиск cmake рядом с VS
    let cmakeExe = findCmake();
    if (!cmakeExe) {
        const common = path.join(installationPath, 'Common7', 'IDE', 'CommonExtensions',
            'Microsoft', 'CMake', 'CMake', 'bin', 'cmake.exe');
        if (fs.existsSync(common)) cmakeExe = common;
    }
    if (!cmakeExe) {
        die('CMake не найден. Обычно он входит в состав Visual Studio VCTools.\n' +
            'Если нет — установите через winget install Kitware.CMake');
    }
    log('CMake:', cmakeExe);

    envSetupCmd = `"${vcvars}"`;
    const cmake = cmakeExe;

    // vcpkg toolchain (опционально)
    const vcpkgToolchain = path.join('C:\\', 'vcpkg', 'scripts', 'buildsystems', 'vcpkg.cmake');
    const vcpkgArg = fs.existsSync(vcpkgToolchain) ? `-DCMAKE_TOOLCHAIN_FILE="${vcpkgToolchain}"` : '';

    // Было: cmake -G "Visual Studio 17 2022" -A x64 ..
    // Стало:
    const cmakeArgs = [
        '-G', 'Visual Studio 17 2022',
        '-A', 'x64',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CUDA_ARCHITECTURES=89',   // RTX 4060 Ti
        '-T', 'cuda=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6',
        '..'
    ];

    const cmd = [
        envSetupCmd,
        `&& "${cmake}" -S "${ENGINE}" -B "${BUILD}" ${cmakeArgs.filter(a => a !== '..').map(a => `"${a}"`).join(' ')} ${vcpkgArg}`,
        `&& "${cmake}" --build "${BUILD}" --config Release --parallel`,
    ].join(' ');

    log('Запуск сборки (Windows)...');
    try {
        execSync(cmd, { stdio: 'inherit' });
    } catch (e) {
        die(`Сборка завершилась с ошибкой: ${e.message}`);
    }
} else {
    // =======================================================================
    // Linux / macOS: gcc/clang + cmake из PATH
    // =======================================================================
    const cmakeExe = findCmake();
    if (!cmakeExe) {
        die('CMake не найден в PATH.\n' +
            'Установите его:\n' +
            '  Debian/Ubuntu: sudo apt install cmake build-essential\n' +
            '  macOS:         brew install cmake');
    }
    log('CMake:', cmakeExe);

    // Проверим компилятор.
    const cc = process.env.CC || (isMac ? 'clang' : 'gcc');
    const cxx = process.env.CXX || (isMac ? 'clang++' : 'g++');
    const ccCheck = spawnSync(isMac ? 'which' : 'which', [cc], { encoding: 'utf8' });
    const cxxCheck = spawnSync(isMac ? 'which' : 'which', [cxx], { encoding: 'utf8' });
    if (!ccCheck.stdout.trim()) {
        die(`Компилятор C '${cc}' не найден. Установите gcc/clang:\n` +
            (isMac
                ? '  xcode-select --install'
                : '  sudo apt install build-essential'));
    }
    if (!cxxCheck.stdout.trim()) {
        die(`Компилятор C++ '${cxx}' не найден.`);
    }
    log(`Компилятор: ${cc} / ${cxx}`);

    const generator = isMac ? '-G "Unix Makefiles"' : '-G "Unix Makefiles"';
    const cmd = [
        `"${cmakeExe}" -S "${ENGINE}" -B "${BUILD}" ${generator}`,
        `&& "${cmakeExe}" --build "${BUILD}" --config Release --parallel`,
    ].join(' ');

    log('Запуск сборки (Unix)...');
    try {
        execSync(cmd, { stdio: 'inherit', shell: '/bin/bash' });
    } catch (e) {
        die(`Сборка завершилась с ошибкой: ${e.message}`);
    }
}

// --- Проверка результата ---
const exeName = isWin ? 'durakk_engine.exe' : 'durakk_engine';
const exeDist = path.join(ENGINE, 'dist', exeName);
const exeRoot = path.join(ROOT, exeName);
log('Готово. Собрано:');
log('  dist :', exeDist, fs.existsSync(exeDist) ? '(есть)' : '(ОТСУТСТВУЕТ)');
log('  root :', exeRoot, fs.existsSync(exeRoot) ? '(есть)' : '(ОТСУТСТВУЕТ)');
