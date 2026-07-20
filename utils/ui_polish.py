from __future__ import annotations

import time

import streamlit as st


PLATFORM_NAME = "NASA Battery Digital Twin"
VERSION = "Version 1.0"
DATASET = "NASA Prognostics Center of Excellence Battery Dataset"


def apply_global_polish() -> None:
    """Apply shared professional styling across the Streamlit application."""

    st.html(
        """
        <style>
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 5.5rem;
        }

        .dt-launch-shell {
            min-height: 70vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 0 1rem;
        }

        .dt-launch-card {
            width: min(1020px, 95%);
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 3.2rem 2.6rem 2.7rem;
            text-align: center;
            border: 1px solid rgba(96, 165, 250, 0.28);
            background:
                radial-gradient(
                    circle at 12% 18%,
                    rgba(59, 130, 246, 0.26),
                    transparent 32%
                ),
                radial-gradient(
                    circle at 88% 78%,
                    rgba(16, 185, 129, 0.18),
                    transparent 31%
                ),
                linear-gradient(
                    145deg,
                    rgba(7, 17, 35, 0.99),
                    rgba(12, 31, 58, 0.99)
                );
            box-shadow:
                0 32px 95px rgba(0, 0, 0, 0.48),
                inset 0 0 0 1px rgba(255, 255, 255, 0.03);
            animation: dtFadeIn 0.8s ease-out both;
        }

        .dt-launch-card::before,
        .dt-launch-card::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            filter: blur(38px);
            pointer-events: none;
        }

        .dt-launch-card::before {
            width: 250px;
            height: 250px;
            top: -110px;
            right: -85px;
            background: rgba(59, 130, 246, 0.30);
            animation: dtGlowPulse 3.2s ease-in-out infinite;
        }

        .dt-launch-card::after {
            width: 220px;
            height: 220px;
            bottom: -105px;
            left: -80px;
            background: rgba(16, 185, 129, 0.22);
            animation: dtGlowPulse 3.9s ease-in-out infinite reverse;
        }

        .dt-title-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin: 0.5rem auto 0.9rem;
            flex-wrap: wrap;
        }

        .dt-title-rocket {
            font-size: 3.6rem;
            line-height: 1;
            animation: dtRocketFloat 2.7s ease-in-out infinite;
            filter: drop-shadow(0 0 12px rgba(96, 165, 250, 0.35));
        }

        .dt-launch-title {
            font-size: clamp(2.3rem, 5vw, 4.2rem);
            line-height: 1.08;
            font-weight: 900;
            letter-spacing: -0.045em;
            color: #f8fafc;
            margin: 0;
            text-shadow: 0 0 25px rgba(96, 165, 250, 0.16);
        }

        .dt-version {
            color: #60a5fa;
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 1.55rem;
        }

        .dt-developed-label {
            color: #cbd5e1;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
        }

        .dt-developer-names {
            color: #ffffff;
            font-size: 1.24rem;
            font-weight: 800;
            line-height: 1.58;
            margin-bottom: 1.35rem;
        }

        .dt-powered {
            color: #bfdbfe;
            font-size: 1.04rem;
            line-height: 1.75;
            max-width: 780px;
            margin: 0 auto;
        }

        .dt-feature-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.72rem;
            margin-top: 1.55rem;
        }

        .dt-feature {
            padding: 0.56rem 0.98rem;
            border-radius: 999px;
            color: #dbeafe;
            font-size: 0.86rem;
            font-weight: 700;
            border: 1px solid rgba(96, 165, 250, 0.28);
            background: rgba(15, 23, 42, 0.62);
            backdrop-filter: blur(6px);
            transition:
                transform 0.18s ease,
                border-color 0.18s ease;
        }

        .dt-feature:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.56);
        }

        .dt-ready-box {
            width: min(600px, 92%);
            margin: 1.8rem auto 0;
            padding: 1rem 1.2rem;
            border-radius: 16px;
            border: 1px solid rgba(34, 197, 94, 0.24);
            background: rgba(20, 83, 45, 0.17);
        }

        .dt-ready-title {
            color: #86efac;
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.28rem;
        }

        .dt-ready-text {
            color: #d1fae5;
            font-size: 0.9rem;
        }

        .dt-ready-dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            margin-right: 7px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 11px rgba(34, 197, 94, 0.85);
        }

        .dt-dataset {
            color: #94a3b8;
            font-size: 0.86rem;
            margin-top: 1.3rem;
        }

        div.stButton > button {
            min-height: 3rem;
            border-radius: 12px;
            font-weight: 800;
            font-size: 0.98rem;
            box-shadow: 0 12px 28px rgba(239, 68, 68, 0.20);
        }

        .dt-footer {
            margin-top: 3rem;
            padding: 1.4rem 1rem;
            border-top: 1px solid rgba(148, 163, 184, 0.20);
            text-align: center;
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.75;
        }

        .dt-footer-title {
            color: #e2e8f0;
            font-weight: 800;
            font-size: 0.98rem;
        }

        .dt-footer-developers {
            color: #e2e8f0;
            font-weight: 700;
            margin: 0.45rem 0;
        }

        @keyframes dtFadeIn {
            from {
                opacity: 0;
                transform: translateY(18px) scale(0.985);
            }

            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        @keyframes dtRocketFloat {
            0%,
            100% {
                transform: translateY(0) rotate(-2deg);
            }

            50% {
                transform: translateY(-9px) rotate(3deg);
            }
        }

        @keyframes dtGlowPulse {
            0%,
            100% {
                transform: scale(0.95);
                opacity: 0.48;
            }

            50% {
                transform: scale(1.08);
                opacity: 0.78;
            }
        }
        </style>
        """
    )


def render_startup_screen() -> None:
    """Show the startup screen and load the platform after button click."""

    if "platform_entered" not in st.session_state:
        st.session_state.platform_entered = False

    if st.session_state.platform_entered:
        return

    startup_html = f"""
    <div class="dt-launch-shell">
        <div class="dt-launch-card">

            <div class="dt-title-wrap">
                <div class="dt-title-rocket">🚀</div>
                <div class="dt-launch-title">{PLATFORM_NAME}</div>
            </div>

            <div class="dt-version">
                {VERSION}
            </div>

            <div class="dt-developed-label">
                Developed by
            </div>

            <div class="dt-developer-names">
                M. Rashid Khan<br>
                &amp;<br>
                Zahid Ullah
            </div>

            <div class="dt-powered">
                Powered by Explainable AI, Digital Twin Technology,
                Predictive Maintenance and Fleet Intelligence.
            </div>

            <div class="dt-feature-row">
                <span class="dt-feature">Digital Twin</span>
                <span class="dt-feature">SOH &amp; RUL</span>
                <span class="dt-feature">Fault Diagnosis</span>
                <span class="dt-feature">Predictive Maintenance</span>
                <span class="dt-feature">Fleet Intelligence</span>
            </div>

            <div class="dt-ready-box">
                <div class="dt-ready-title">
                    <span class="dt-ready-dot"></span>
                    Mission Control Ready
                </div>

                <div class="dt-ready-text">
                    Click “Launch Command Center” below to enter the platform.
                </div>
            </div>

            <div class="dt-dataset">
                Data source: {DATASET}
            </div>

        </div>
    </div>
    """

    st.html(startup_html)

    left, center, right = st.columns([1.35, 1.0, 1.35])

    with center:
        if st.button(
            "🚀 Launch Command Center",
            use_container_width=True,
            type="primary",
        ):
            status = st.empty()
            progress = st.progress(0)

            launch_steps = [
                ("Initializing AI engine...", 20),
                ("Connecting digital twin models...", 40),
                ("Loading fleet telemetry...", 60),
                ("Starting predictive maintenance...", 80),
                ("Opening mission command center...", 100),
            ]

            for message, percentage in launch_steps:
                status.info(message)
                progress.progress(percentage)
                time.sleep(0.5)

            status.success("Platform ready.")
            time.sleep(0.4)

            st.session_state.platform_entered = True
            st.rerun()

    st.stop()


def render_footer() -> None:
    """Render the shared professional footer."""

    footer_html = f"""
    <div class="dt-footer">
        <div class="dt-footer-title">
            {PLATFORM_NAME} v1.0
        </div>

        <div style="margin-top: 0.55rem;">
            Developed by
        </div>

        <div class="dt-footer-developers">
            M. Rashid Khan<br>
            &amp;<br>
            Zahid Ullah
        </div>

        Powered by Explainable AI and Digital Twin Technology<br>
        Data source: {DATASET}<br>
        © 2026 Research Project • All Rights Reserved
    </div>
    """

    st.html(footer_html)
