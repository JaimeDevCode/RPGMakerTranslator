//=============================================================================
// TranslationCrashLogger.js
//=============================================================================
/*:
 * @plugindesc [v1.0] Crash Logger — writes uncaught errors to crash_log.txt
 * @author RPGMakerTranslator
 *
 * @help
 * This plugin captures all uncaught JavaScript errors and writes them to
 * a file called "crash_log.txt" in the game's root directory.
 *
 * It hooks into:
 *   - window.onerror  (browser-level JS errors)
 *   - window.onunhandledrejection  (unhandled Promise rejections)
 *   - process.on('uncaughtException')  (Node.js-level fatal errors)
 *
 * On NW.js (desktop), errors are written to disk and an alert is shown.
 * On browser, errors are logged to console and shown via alert.
 *
 * Place this plugin at the TOP of the plugin list so it loads first.
 *
 * @param ShowAlert
 * @desc Show an alert dialog when an error occurs?
 * @type boolean
 * @default true
 *
 * @param PreventClose
 * @desc Try to prevent the game from closing on error?
 * @type boolean
 * @default true
 */

(function() {
    'use strict';

    var params = PluginManager.parameters('TranslationCrashLogger');
    var showAlert = params['ShowAlert'] !== 'false';
    var preventClose = params['PreventClose'] !== 'false';

    // Detect environment
    var isNwJs = typeof require === 'function' && typeof process === 'object';
    var fs = null;
    var path = null;
    var logPath = null;

    if (isNwJs) {
        try {
            fs = require('fs');
            path = require('path');
            // Write to the same directory as the executable
            var gameDir = path.dirname(process.execPath);
            logPath = path.join(gameDir, 'crash_log.txt');
        } catch (e) {
            console.error('[CrashLogger] Failed to initialize fs:', e);
        }
    }

    function timestamp() {
        var d = new Date();
        return d.getFullYear() + '-' +
               String(d.getMonth() + 1).padStart(2, '0') + '-' +
               String(d.getDate()).padStart(2, '0') + ' ' +
               String(d.getHours()).padStart(2, '0') + ':' +
               String(d.getMinutes()).padStart(2, '0') + ':' +
               String(d.getSeconds()).padStart(2, '0') + '.' +
               String(d.getMilliseconds()).padStart(3, '0');
    }

    function writeLog(text) {
        var entry = '[' + timestamp() + '] ' + text + '\n\n';

        // Always log to console
        console.error('[CrashLogger]', text);

        // Write to file if in NW.js
        if (fs && logPath) {
            try {
                fs.appendFileSync(logPath, entry, 'utf8');
            } catch (e) {
                console.error('[CrashLogger] Failed to write log:', e);
            }
        }

        // Show alert
        if (showAlert) {
            try {
                alert('[CrashLogger]\n' + text + '\n\nSee crash_log.txt for details.');
            } catch (e) {
                // alert might fail in some contexts
            }
        }
    }

    // Log startup
    var startMsg = '=== Game Started ===';
    if (isNwJs) {
        startMsg += '\nNW.js: ' + (process.versions['nw'] || 'unknown');
        startMsg += '\nChromium: ' + (process.versions['chromium'] || 'unknown');
        startMsg += '\nNode: ' + (process.versions['node'] || 'unknown');
    }
    writeLog(startMsg);

    // 1) window.onerror — catches runtime JS errors
    var _oldOnError = window.onerror;
    window.onerror = function(message, source, lineno, colno, error) {
        var info = 'UNCAUGHT ERROR\n';
        info += 'Message: ' + message + '\n';
        info += 'Source: ' + source + '\n';
        info += 'Location: Line ' + lineno + ', Col ' + colno + '\n';
        if (error && error.stack) {
            info += 'Stack:\n' + error.stack;
        }
        writeLog(info);
        if (_oldOnError) {
            _oldOnError.apply(this, arguments);
        }
        return preventClose; // true = suppress default handling
    };

    // 2) Unhandled promise rejections
    // Known harmless patterns that should be silently ignored
    var _ignorePatterns = [
        /Failed to fetch/i,
        /Load timeout/i,
        /NetworkError/i,
        /Cheat\.html/i
    ];

    function _isIgnored(msg) {
        for (var i = 0; i < _ignorePatterns.length; i++) {
            if (_ignorePatterns[i].test(msg)) return true;
        }
        return false;
    }

    window.addEventListener('unhandledrejection', function(event) {
        var reason = event.reason;
        var msg = (reason instanceof Error) ? reason.message : String(reason);

        // Silently swallow known harmless rejections (e.g. Live2D fetch)
        if (_isIgnored(msg)) {
            event.preventDefault();
            return;
        }

        var info = 'UNHANDLED PROMISE REJECTION\n';
        if (reason instanceof Error) {
            info += 'Message: ' + reason.message + '\n';
            if (reason.stack) info += 'Stack:\n' + reason.stack;
        } else {
            info += 'Reason: ' + msg;
        }
        writeLog(info);
        if (preventClose) {
            event.preventDefault();
        }
    });

    // 3) Node.js uncaughtException — catches errors that window.onerror misses
    if (isNwJs && process && process.on) {
        process.on('uncaughtException', function(error) {
            var msg = error.message || '';
            var stack = error.stack || '';
            if (_isIgnored(msg) || _isIgnored(stack)) return;
            var info = 'NODE UNCAUGHT EXCEPTION\n';
            info += 'Message: ' + msg + '\n';
            if (stack) info += 'Stack:\n' + stack;
            writeLog(info);
        });
    }

    // 4) Override SceneManager.catchException to also log
    var _Scene_catchException = null;
    if (typeof SceneManager !== 'undefined') {
        _Scene_catchException = SceneManager.catchException;
        SceneManager.catchException = function(e) {
            var info = 'SceneManager.catchException\n';
            if (e instanceof Error) {
                info += 'Message: ' + e.message + '\n';
                if (e.stack) info += 'Stack:\n' + e.stack;
            } else {
                info += 'Error: ' + String(e);
            }
            writeLog(info);
            if (_Scene_catchException) {
                _Scene_catchException.call(this, e);
            }
        };
    }

    // 5) Override SceneManager.onError to also log (MZ)
    if (typeof SceneManager !== 'undefined' && SceneManager.onError) {
        var _Scene_onError = SceneManager.onError;
        SceneManager.onError = function(e) {
            var info = 'SceneManager.onError\n';
            if (e instanceof Error) {
                info += 'Message: ' + e.message + '\n';
                if (e.stack) info += 'Stack:\n' + e.stack;
            } else if (e && e.message) {
                info += 'Message: ' + e.message + '\n';
                info += 'Filename: ' + (e.filename || '') + '\n';
                info += 'Location: Line ' + (e.lineno || '') + '\n';
            } else {
                info += 'Error: ' + String(e);
            }
            writeLog(info);
            if (_Scene_onError) {
                _Scene_onError.call(this, e);
            }
        };
    }

    // Log successful plugin initialization
    writeLog('CrashLogger initialized successfully. Monitoring for errors...');

})();
