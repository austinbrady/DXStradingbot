/**
 * DXS HFT Bot - Toolbar Injector
 * Injects control toolbar into DXS.app page
 * Works with existing authenticated session - NO CREDENTIALS NEEDED
 */

(function() {
    'use strict';

    // Create toolbar HTML
    const toolbarHTML = `
    <div id="hft-toolbar" class="hft-toolbar">
        <!-- Left Section: Status & Controls -->
        <div class="toolbar-section toolbar-left">
            <div class="status-indicator">
                <div class="status-light" id="status-light"></div>
                <span id="status-text">STOPPED</span>
            </div>
            
            <div class="control-buttons">
                <button id="btn-start" class="btn btn-start" title="Start Trading">▶️ START</button>
                <button id="btn-pause" class="btn btn-pause" title="Pause Trading" disabled>⏸️ PAUSE</button>
                <button id="btn-resume" class="btn btn-resume" title="Resume Trading" disabled>▶️ RESUME</button>
                <button id="btn-stop" class="btn btn-stop" title="Stop Trading" disabled>⏹️ STOP</button>
                <button id="btn-emergency" class="btn btn-emergency" title="Emergency Stop">🚨 EMERGENCY</button>
            </div>
        </div>

        <!-- Center Section: Metrics -->
        <div class="toolbar-section toolbar-center">
            <div class="metric">
                <span class="metric-label">TRADES</span>
                <span class="metric-value" id="metric-trades">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">WIN RATE</span>
                <span class="metric-value" id="metric-winrate">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">PnL</span>
                <span class="metric-value metric-pnl" id="metric-pnl">$0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">ACTIVE</span>
                <span class="metric-value" id="metric-active">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">UPTIME</span>
                <span class="metric-value" id="metric-uptime">0s</span>
            </div>
        </div>

        <!-- Right Section: Settings -->
        <div class="toolbar-section toolbar-right">
            <div class="setting">
                <label for="setting-size">Position Size ($)</label>
                <input type="number" id="setting-size" value="5.0" min="0.1" max="100" step="0.1">
            </div>
            <div class="setting">
                <label for="setting-leverage">Max Leverage</label>
                <input type="number" id="setting-leverage" value="20" min="1" max="40" step="1">
            </div>
            <button id="btn-settings" class="btn btn-settings" title="Advanced Settings">⚙️</button>
            <button id="btn-minimize" class="btn btn-minimize" title="Minimize">−</button>
        </div>
    </div>

    <!-- Settings Modal -->
    <div id="settings-modal" class="settings-modal" style="display:none;">
        <div class="settings-content">
            <h3>Bot Settings</h3>
            <div class="settings-grid">
                <div class="setting-item">
                    <label>Base Position Size (USD)</label>
                    <input type="number" id="set-base-size" value="5.0" min="0.1" max="100">
                </div>
                <div class="setting-item">
                    <label>Max Leverage</label>
                    <input type="number" id="set-max-leverage" value="20" min="1" max="40">
                </div>
                <div class="setting-item">
                    <label>Stop Loss %</label>
                    <input type="number" id="set-stop-loss" value="0.15" min="0.05" max="1.0" step="0.05">
                </div>
                <div class="setting-item">
                    <label>Take Profit %</label>
                    <input type="number" id="set-take-profit" value="0.10" min="0.05" max="0.5" step="0.05">
                </div>
                <div class="setting-item">
                    <label>Check Interval (ms)</label>
                    <input type="number" id="set-interval" value="100" min="50" max="1000">
                </div>
                <div class="setting-item">
                    <label>Trading Symbols (comma-separated)</label>
                    <input type="text" id="set-symbols" value="BTC,ETH,BNB,XRP,ADA,SOL">
                </div>
            </div>
            <div class="settings-buttons">
                <button id="btn-save-settings" class="btn btn-primary">Save Settings</button>
                <button id="btn-close-settings" class="btn btn-secondary">Close</button>
            </div>
        </div>
    </div>

    <!-- Trades Log Modal -->
    <div id="trades-modal" class="trades-modal" style="display:none;">
        <div class="trades-content">
            <h3>Recent Trades</h3>
            <div id="trades-list" class="trades-list"></div>
            <button id="btn-close-trades" class="btn btn-secondary">Close</button>
        </div>
    </div>

    <style>
        /* Toolbar Styles */
        .hft-toolbar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 90px;
            background: linear-gradient(to top, #0a0e27, #1a1f3a);
            border-top: 2px solid #00d9ff;
            box-shadow: 0 -5px 30px rgba(0, 217, 255, 0.2);
            z-index: 9999;
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: #00d9ff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            gap: 30px;
        }

        .toolbar-section {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }

        .toolbar-left {
            flex: 1;
            min-width: 400px;
        }

        .toolbar-center {
            flex: 2;
            justify-content: center;
            gap: 25px;
        }

        .toolbar-right {
            flex: 1;
            justify-content: flex-end;
            gap: 10px;
        }

        /* Status Indicator */
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding-right: 15px;
            border-right: 1px solid #00d9ff;
        }

        .status-light {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-light.stopped { 
            background-color: #ff4444; 
            animation: none;
        }

        .status-light.initializing { 
            background-color: #ffaa00; 
            animation: pulse 1s infinite; 
        }

        .status-light.running { 
            background-color: #00ff88; 
            animation: pulse 0.5s infinite; 
        }

        .status-light.paused { 
            background-color: #ffaa00;
            animation: pulse 1.5s infinite;
        }

        .status-light.emergency { 
            background-color: #ff2200; 
            animation: pulse 0.3s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Control Buttons */
        .control-buttons {
            display: flex;
            gap: 8px;
        }

        .btn {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: bold;
            transition: all 0.2s;
            white-space: nowrap;
            background: #1a3a52;
            color: #00d9ff;
            border: 1px solid #00d9ff;
        }

        .btn:hover:not(:disabled) {
            background: #00d9ff;
            color: #0a0e27;
            transform: translateY(-1px);
            box-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
        }

        .btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .btn-start {
            background: #004d00;
            border-color: #00ff00;
            color: #00ff00;
        }

        .btn-start:hover { 
            background: #00ff00;
            color: #004d00;
        }

        .btn-stop {
            background: #4d0000;
            border-color: #ff4444;
            color: #ff4444;
        }

        .btn-stop:hover { 
            background: #ff4444;
            color: #4d0000;
        }

        .btn-emergency {
            background: #ff2200;
            border-color: #ff2200;
            color: white;
            animation: pulse-urgent 0.5s infinite;
        }

        @keyframes pulse-urgent {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .btn-emergency:hover {
            background: white;
            color: #ff2200;
        }

        /* Metrics */
        .metric {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 3px;
            padding: 0 10px;
        }

        .metric-label {
            font-size: 10px;
            opacity: 0.7;
        }

        .metric-value {
            font-size: 14px;
            font-weight: bold;
            color: #00ff88;
        }

        .metric-pnl {
            color: #00d9ff;
        }

        .metric-pnl.positive {
            color: #00ff88;
        }

        .metric-pnl.negative {
            color: #ff4444;
        }

        /* Settings */
        .setting {
            display: flex;
            flex-direction: column;
            gap: 2px;
            align-items: flex-end;
        }

        .setting label {
            font-size: 9px;
            opacity: 0.7;
        }

        .setting input {
            width: 60px;
            padding: 3px 5px;
            background: #0a0e27;
            border: 1px solid #00d9ff;
            color: #00d9ff;
            border-radius: 2px;
            font-size: 11px;
        }

        .setting input:focus {
            outline: none;
            box-shadow: 0 0 5px rgba(0, 217, 255, 0.5);
        }

        .btn-settings, .btn-minimize {
            padding: 6px 8px;
        }

        /* Modals */
        .settings-modal, .trades-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .settings-content, .trades-content {
            background: linear-gradient(to bottom, #1a1f3a, #0a0e27);
            border: 2px solid #00d9ff;
            border-radius: 8px;
            padding: 25px;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
        }

        .settings-content h3, .trades-content h3 {
            color: #00d9ff;
            margin-bottom: 15px;
            font-size: 18px;
        }

        .settings-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }

        .setting-item label {
            display: block;
            color: #00d9ff;
            font-size: 12px;
            margin-bottom: 5px;
        }

        .setting-item input {
            width: 100%;
            padding: 8px;
            background: #0a0e27;
            border: 1px solid #00d9ff;
            color: #00ff88;
            border-radius: 4px;
            font-size: 12px;
        }

        .settings-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }

        .btn-primary {
            background: #00d9ff;
            color: #0a0e27;
        }

        .btn-secondary {
            background: #1a3a52;
            color: #00d9ff;
            border: 1px solid #00d9ff;
        }

        .trades-list {
            background: #0a0e27;
            border: 1px solid #00d9ff;
            border-radius: 4px;
            max-height: 50vh;
            overflow-y: auto;
            margin-bottom: 15px;
        }

        .trade-item {
            padding: 10px;
            border-bottom: 1px solid #1a3a52;
            font-size: 11px;
            color: #00d9ff;
        }

        .trade-item.win {
            border-left: 3px solid #00ff88;
        }

        .trade-item.loss {
            border-left: 3px solid #ff4444;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .hft-toolbar {
                height: auto;
                flex-direction: column;
                padding: 10px;
                gap: 10px;
            }

            .toolbar-section {
                width: 100%;
            }

            .toolbar-center {
                justify-content: space-around;
            }
        }
    </style>
    `;

    // State management
    const botState = {
        state: 'STOPPED',
        metrics: {
            trades_executed: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_pnl: 0,
            win_rate: 0,
            active_positions: 0,
            uptime_seconds: 0
        },
        settings: {
            baseSize: 5.0,
            maxLeverage: 20,
            symbols: ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL']
        }
    };

    // Initialize toolbar
    function initToolbar() {
        // Inject HTML
        const toolbarContainer = document.createElement('div');
        toolbarContainer.innerHTML = toolbarHTML;
        document.body.appendChild(toolbarContainer.firstElementChild);

        // Add bottom padding to main content
        document.documentElement.style.paddingBottom = '100px';

        // Attach event listeners
        attachEventListeners();

        // Start metrics update loop
        updateMetricsDisplay();
        setInterval(updateMetricsDisplay, 1000);

        console.log('🤖 DXS HFT Bot Toolbar Injected');
    }

    // Event listeners
    function attachEventListeners() {
        document.getElementById('btn-start').addEventListener('click', () => {
            botState.state = 'RUNNING';
            updateToolbarUI();
            console.log('▶️ Bot started');
        });

        document.getElementById('btn-pause').addEventListener('click', () => {
            botState.state = 'PAUSED';
            updateToolbarUI();
            console.log('⏸️ Bot paused');
        });

        document.getElementById('btn-resume').addEventListener('click', () => {
            botState.state = 'RUNNING';
            updateToolbarUI();
            console.log('▶️ Bot resumed');
        });

        document.getElementById('btn-stop').addEventListener('click', () => {
            botState.state = 'STOPPED';
            updateToolbarUI();
            console.log('⏹️ Bot stopped');
        });

        document.getElementById('btn-emergency').addEventListener('click', () => {
            if (confirm('🚨 EMERGENCY STOP - Are you sure?')) {
                botState.state = 'EMERGENCY_STOP';
                updateToolbarUI();
                console.log('🚨 EMERGENCY STOP ACTIVATED');
            }
        });

        document.getElementById('btn-settings').addEventListener('click', () => {
            document.getElementById('settings-modal').style.display = 'flex';
        });

        document.getElementById('btn-close-settings').addEventListener('click', () => {
            document.getElementById('settings-modal').style.display = 'none';
        });

        document.getElementById('btn-save-settings').addEventListener('click', () => {
            botState.settings.baseSize = parseFloat(document.getElementById('set-base-size').value);
            botState.settings.maxLeverage = parseInt(document.getElementById('set-max-leverage').value);
            botState.settings.symbols = document.getElementById('set-symbols').value.split(',').map(s => s.trim());
            
            document.getElementById('settings-modal').style.display = 'none';
            console.log('✅ Settings saved', botState.settings);
        });

        document.getElementById('btn-minimize').addEventListener('click', () => {
            const toolbar = document.getElementById('hft-toolbar');
            toolbar.style.height = toolbar.style.height === '30px' ? '90px' : '30px';
        });
    }

    // Update toolbar UI
    function updateToolbarUI() {
        const state = botState.state;
        const startBtn = document.getElementById('btn-start');
        const pauseBtn = document.getElementById('btn-pause');
        const resumeBtn = document.getElementById('btn-resume');
        const stopBtn = document.getElementById('btn-stop');
        const statusText = document.getElementById('status-text');
        const statusLight = document.getElementById('status-light');

        // Reset all
        startBtn.disabled = false;
        pauseBtn.disabled = true;
        resumeBtn.disabled = true;
        stopBtn.disabled = true;

        // Update based on state
        statusText.textContent = state;
        statusLight.className = 'status-light ' + state.toLowerCase();

        if (state === 'RUNNING') {
            pauseBtn.disabled = false;
            stopBtn.disabled = false;
            startBtn.disabled = true;
        } else if (state === 'PAUSED') {
            resumeBtn.disabled = false;
            stopBtn.disabled = false;
        } else if (state === 'STOPPED' || state === 'EMERGENCY_STOP') {
            // All buttons ready
        }
    }

    // Update metrics display
    function updateMetricsDisplay() {
        // Simulate metrics update (in real implementation, these come from bot)
        document.getElementById('metric-trades').textContent = botState.metrics.trades_executed;
        document.getElementById('metric-winrate').textContent = botState.metrics.win_rate.toFixed(1) + '%';
        
        const pnlElement = document.getElementById('metric-pnl');
        const pnlValue = botState.metrics.total_pnl;
        pnlElement.textContent = '$' + pnlValue.toFixed(2);
        pnlElement.className = 'metric-value metric-pnl ' + (pnlValue >= 0 ? 'positive' : 'negative');
        
        document.getElementById('metric-active').textContent = botState.metrics.active_positions;
        document.getElementById('metric-uptime').textContent = botState.metrics.uptime_seconds + 's';
    }