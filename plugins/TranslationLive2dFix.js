//=============================================================================
// TranslationLive2dFix.js
//=============================================================================
/*:
 * @plugindesc [v1.3] Null-guard + DRM bypass for enc_lv2d.js
 * @author RPGMakerTranslator
 *
 * @help
 * enc_lv2d.js overrides several SceneManager and Scene_Base methods to call
 * $gameLive2d methods.  However, $gameLive2d is null until
 * DataManager.createGameObjects() runs (on New Game / Load Game).
 * Scene lifecycle hooks fire during Scene_Boot and Scene_Title, BEFORE
 * the variable is initialised, causing "Cannot read property ... of null".
 *
 * enc_lv2d.js also contains DRM checks that compare data values (e.g.
 * weapon names) against hardcoded originals.  Translation changes those
 * values, so the DRM would call SceneManager.exit().  This plugin
 * neutralises those checks so the translated game keeps running.
 *
 * This plugin MUST load BEFORE enc_lv2d.js.
 *
 *   Phase 1 (plugin load):
 *     Saves references to the original RPG Maker methods before any plugin
 *     overrides them.
 *
 *   Phase 2 (SceneManager.run, after ALL plugins loaded):
 *     - Wraps every enc_lv2d-overridden method with a $gameLive2d null check.
 *     - Wraps DataManager.onLoad and Scene_Title.start to suppress
 *       DRM-triggered SceneManager.exit() calls.
 *
 * @param Enable
 * @type boolean
 * @default true
 */

(function() {
    'use strict';

    var params = PluginManager.parameters('TranslationLive2dFix');
    if (params['Enable'] === 'false') return;

    // -----------------------------------------------------------------------
    //  Phase 1 – Save original RPG Maker methods before enc_lv2d.js loads
    // -----------------------------------------------------------------------
    var _saved = {};

    // SceneManager (defined in rpg_managers.js, always present)
    _saved.SM_changeScene        = SceneManager.changeScene;
    _saved.SM_snapForBackground  = SceneManager.snapForBackground;
    _saved.SM_onSceneCreate      = SceneManager.onSceneCreate;

    // Scene_Base (defined in rpg_scenes.js)
    _saved.SB_update    = Scene_Base.prototype.update;
    _saved.SB_terminate = Scene_Base.prototype.terminate;
    _saved.SB_create    = Scene_Base.prototype.create;

    // Optional classes – guard with typeof
    if (typeof Scene_GameEnd !== 'undefined')
        _saved.SGE_commandToTitle = Scene_GameEnd.prototype.commandToTitle;
    if (typeof Game_Interpreter !== 'undefined')
        _saved.GI_command354 = Game_Interpreter.prototype.command354;
    if (typeof Scene_Load !== 'undefined') {
        _saved.SL_onSavefileOk  = Scene_Load.prototype.onSavefileOk;
        _saved.SL_onLoadSuccess = Scene_Load.prototype.onLoadSuccess;
    }
    if (typeof Scene_Save !== 'undefined')
        _saved.SS_onSavefileOk = Scene_Save.prototype.onSavefileOk;
    if (typeof Spriteset_Base !== 'undefined')
        _saved.SpB_update = Spriteset_Base.prototype.update;
    if (typeof Scene_MenuBase !== 'undefined')
        _saved.SMB_createBackground = Scene_MenuBase.prototype.createBackground;

    // -----------------------------------------------------------------------
    //  Phase 2 – Hook SceneManager.run (called AFTER all plugins loaded)
    // -----------------------------------------------------------------------
    var _orig_SM_run = SceneManager.run;
    SceneManager.run = function(sceneClass) {
        // At this point every plugin IIFE has executed.
        // If enc_lv2d.js declared $gameLive2d, apply our wraps.
        if (typeof $gameLive2d !== 'undefined') {
            _applyNullGuards();
        }
        return _orig_SM_run.apply(this, arguments);
    };

    // -----------------------------------------------------------------------
    //  Wrap helpers
    // -----------------------------------------------------------------------
    /**
     * Wrap a direct method on `obj`.
     * When $gameLive2d is falsy, call the Phase-1 `fallback` instead.
     * If the enc_lv2d override throws (e.g. null internal models),
     * catch the error and fall back to the original RPG Maker method.
     */
    function _wrapMethod(obj, name, fallback) {
        var current = obj[name];
        if (!current || current === fallback) return;
        obj[name] = function() {
            if (typeof $gameLive2d === 'undefined' || !$gameLive2d) {
                return fallback.apply(this, arguments);
            }
            try {
                return current.apply(this, arguments);
            } catch (e) {
                console.warn('[Live2dFix] ' + name + ' error, using fallback:', e.message);
                return fallback.apply(this, arguments);
            }
        };
    }

    /**
     * Same as _wrapMethod but operates on `cls.prototype[name]`.
     */
    function _wrapProto(cls, name, fallback) {
        if (!cls) return;
        var current = cls.prototype[name];
        if (!current || current === fallback) return;
        cls.prototype[name] = function() {
            if (typeof $gameLive2d === 'undefined' || !$gameLive2d) {
                return fallback.apply(this, arguments);
            }
            try {
                return current.apply(this, arguments);
            } catch (e) {
                console.warn('[Live2dFix] ' + name + ' error, using fallback:', e.message);
                return fallback.apply(this, arguments);
            }
        };
    }

    // -----------------------------------------------------------------------
    //  Apply all wraps
    // -----------------------------------------------------------------------
    function _applyNullGuards() {
        // SceneManager methods
        _wrapMethod(SceneManager, 'changeScene',       _saved.SM_changeScene);
        _wrapMethod(SceneManager, 'snapForBackground',  _saved.SM_snapForBackground);
        _wrapMethod(SceneManager, 'onSceneCreate',      _saved.SM_onSceneCreate);

        // Scene_Base prototype
        _wrapProto(Scene_Base, 'update',    _saved.SB_update);
        _wrapProto(Scene_Base, 'terminate', _saved.SB_terminate);
        _wrapProto(Scene_Base, 'create',    _saved.SB_create);

        // Scene_GameEnd / Game_Interpreter
        if (_saved.SGE_commandToTitle)
            _wrapProto(Scene_GameEnd, 'commandToTitle', _saved.SGE_commandToTitle);
        if (_saved.GI_command354)
            _wrapProto(Game_Interpreter, 'command354', _saved.GI_command354);

        // Scene_Load
        if (_saved.SL_onSavefileOk)
            _wrapProto(Scene_Load, 'onSavefileOk', _saved.SL_onSavefileOk);
        if (_saved.SL_onLoadSuccess)
            _wrapProto(Scene_Load, 'onLoadSuccess', _saved.SL_onLoadSuccess);

        // Scene_Save
        if (_saved.SS_onSavefileOk)
            _wrapProto(Scene_Save, 'onSavefileOk', _saved.SS_onSavefileOk);

        // Spriteset_Base
        if (_saved.SpB_update)
            _wrapProto(Spriteset_Base, 'update', _saved.SpB_update);

        // Scene_MenuBase.createBackground
        if (_saved.SMB_createBackground)
            _wrapProto(Scene_MenuBase, 'createBackground', _saved.SMB_createBackground);

        // -------------------------------------------------------------------
        //  DRM neutralisation
        // -------------------------------------------------------------------
        // enc_lv2d.js has integrity checks that call SceneManager.exit()
        // when translated data doesn't match the original Japanese values:
        //   1) DataManager.onLoad: $dataWeapons[4].name != '皮の弓' → exit
        //   2) Scene_Title.start: Graphics._prop != 'pPvdiii63h'   → exit
        //   3) SceneManager.run: file existence check (unaffected by translation)
        //
        // We suppress SceneManager.exit() during these specific callbacks
        // so the translated game keeps running.

        // Wrap DataManager.onLoad — suppress exit during weapon name check
        var _dm_onLoad = DataManager.onLoad;
        if (_dm_onLoad) {
            DataManager.onLoad = function(obj) {
                var realExit = SceneManager.exit;
                SceneManager.exit = function() { /* DRM suppressed */ };
                try {
                    return _dm_onLoad.apply(this, arguments);
                } finally {
                    SceneManager.exit = realExit;
                }
            };
        }

        // Wrap Scene_Title.start — suppress exit during Graphics check
        if (typeof Scene_Title !== 'undefined') {
            var _st_start = Scene_Title.prototype.start;
            if (_st_start) {
                Scene_Title.prototype.start = function() {
                    var realExit = SceneManager.exit;
                    SceneManager.exit = function() { /* DRM suppressed */ };
                    try {
                        return _st_start.apply(this, arguments);
                    } finally {
                        SceneManager.exit = realExit;
                    }
                };
            }
        }
    }

})();
