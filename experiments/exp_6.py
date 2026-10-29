import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

# =============================================================================
# 1. PARAMETERS (Exact match to paper: n=10, dsafe=1.0, Rc=6.0, beta=0.5)
# =============================================================================
n = 10                  # Number of agents
d = 2                   # Dimension (R^2)
beta = 0.5              # Fractional power velocity exponent
d_safe = 1.0            # Pairwise safety threshold [m]
R_c = 6.0               # Communication cutoff range [m]
eps_w = 0.05            # Weight regularization constant
nu = 1.0                # Range-dependent weight exponent
w_under = 1.0 / (R_c**nu + eps_w)  # Uniform lower bound on active weights (~0.1653)

m_gain = 10.0           # Cooperative coupling gain M = m * I_2
M_mat = m_gain * np.eye(d)

# Corridor geometry
x_corr_start = 12.0
x_corr_end = 30.0
y_wall = 1.90           # Half-width (total corridor width = 3.80 m)
wall_height = 4.20      # Thickness for visualization

v_ref = 2.50            # Swarm reference speed [m/s]
t_final = 17.5          # Total maneuver duration [s]

# =============================================================================
# 2. SPANNING BACKBONE GRAPH H (10-cycle from user sketch)
# =============================================================================
# Agent labeling (0-indexed internally: Agent 1 -> idx 0, Agent 10 -> idx 9):
# 0: Agent 1 (lower rear notch)
# 1: Agent 2 (lower rear corner)
# 2: Agent 3 (lower mid wing)
# 3: Agent 4 (lower front wing)
# 4: Agent 5 (Apex / Leader at front)
# 5: Agent 6 (upper front wing)
# 6: Agent 7 (upper mid wing)
# 7: Agent 8 (upper rear corner)
# 8: Agent 9 (upper rear notch)
# 9: Agent 10 (center notch apex)

backbone_edges = [
    (0, 1),  # 1 - 2
    (1, 2),  # 2 - 3
    (2, 3),  # 3 - 4
    (3, 4),  # 4 - 5
    (4, 5),  # 5 - 6
    (5, 6),  # 6 - 7
    (6, 7),  # 7 - 8
    (7, 8),  # 8 - 9
    (8, 9),  # 9 - 10
    (9, 0)   # 10 - 1
]

# Build unweighted backbone Laplacian L_H
L_H = np.zeros((n, n))
for i, j in backbone_edges:
    L_H[i, i] += 1
    L_H[j, j] += 1
    L_H[i, j] -= 1
    L_H[j, i] -= 1

eigvals_H = eigh(L_H, eigvals_only=True)
lambda_2_H = eigvals_H[1]  # Exactly 2 - 2*cos(36 deg) = 0.3820
gamma_H = 1.0 + m_gain * w_under * lambda_2_H

# =============================================================================
# 3. FORMATION TEMPLATES: RING (OUTSIDE) & CHEVRON (INSIDE)
# =============================================================================
def get_base_templates():
    # Template A: Regular Decagon Ring (radius R = 3.60 m)
    # Apex (Agent 5) at theta = 0 deg
    R_ring = 3.60
    theta = np.zeros(n)
    for idx in range(n):
        agent_num = idx + 1
        # 5 -> 0 deg, 6 -> 36 deg, ..., 10 -> 180 deg, 1 -> 216 deg, ..., 4 -> 324 deg
        theta[idx] = np.deg2rad((agent_num - 5) * 36.0)
    
    P_ring = np.column_stack([R_ring * np.cos(theta), R_ring * np.sin(theta)])
    
    # Template B: Chevron inside corridor (fits within y in [-1.50, 1.50] m)
    # Preserves the identical 10-cycle topology with > 1.4 m pairwise distances
    P_chev_raw = np.array([
        [-1.20, -0.70],  # 1: Agent 1 (lower rear notch)
        [ 0.00, -1.50],  # 2: Agent 2 (lower rear wing tip)
        [ 2.00, -1.10],  # 3: Agent 3 (lower mid wing)
        [ 4.00, -0.70],  # 4: Agent 4 (lower front wing)
        [ 5.80,  0.00],  # 5: Agent 5 (Apex / Leader)
        [ 4.00,  0.70],  # 6: Agent 6 (upper front wing)
        [ 2.00,  1.10],  # 7: Agent 7 (upper mid wing)
        [ 0.00,  1.50],  # 8: Agent 8 (upper rear wing tip)
        [-1.20,  0.70],  # 9: Agent 9 (upper rear notch)
        [ 0.10,  0.00]   # 10: Agent 10 (center notch tip)
    ])
    
    # Center offsets to satisfy sum_{i=1}^n p_i = 0 (Paper Eq. 16)
    P_ring_centered = P_ring - np.mean(P_ring, axis=0, keepdims=True)
    P_chev_centered = P_chev_raw - np.mean(P_chev_raw, axis=0, keepdims=True)
    return P_ring_centered, P_chev_centered

P_RING, P_CHEV = get_base_templates()
DP_TEMPLATE = P_CHEV - P_RING

def quintic_step(tau):
    """C^2 smooth quintic step function on tau in [0, 1]."""
    tau_c = np.clip(tau, 0.0, 1.0)
    s = 10.0 * tau_c**3 - 15.0 * tau_c**4 + 6.0 * tau_c**5
    ds = (30.0 * tau_c**2 - 60.0 * tau_c**3 + 30.0 * tau_c**4) * ((tau >= 0.0) & (tau <= 1.0))
    return s, ds

def morph_schedule(t):
    """
    alpha(t) in [0, 1]:
      0.0 -> Ring Formation
      1.0 -> Chevron Formation
    
    Schedule:
    - [0.0, 1.2] s  : Hold Ring (Finite-time convergence T ~ 1.1 s completes)
    - [1.2, 2.8] s  : Morph Ring -> Chevron (completes before apex reaches corridor at x=12 m)
    - [2.8, 13.5] s : Maintain Chevron throughout corridor
                      (Rear agents 1 & 9 clear corridor at t = 13.1 s)
    - [13.5, 15.5] s: Morph Chevron -> Ring (occurs AFTER last agent exits)
    - [15.5, 17.5] s: Hold Ring
    """
    t1_s, t1_e = 1.2, 2.8
    t2_s, t2_e = 13.5, 15.5

    if t < t1_s:
        return 0.0, 0.0
    elif t < t1_e:
        s, ds = quintic_step((t - t1_s) / (t1_e - t1_s))
        return s, ds / (t1_e - t1_s)
    elif t < t2_s:
        return 1.0, 0.0
    elif t < t2_e:
        s, ds = quintic_step((t - t2_s) / (t2_e - t2_s))
        return 1.0 - s, -ds / (t2_e - t2_s)
    else:
        return 0.0, 0.0

def get_stations_and_derivatives(t):
    r = np.array([v_ref * t, 0.0])
    dr = np.array([v_ref, 0.0])
    alpha, dalpha = morph_schedule(t)
    p = P_RING + alpha * DP_TEMPLATE
    dp = dalpha * DP_TEMPLATE
    q = r + p
    dq = dr + dp
    return q, dq, r, dr

# =============================================================================
# 4. INITIALIZATION & CONTROLLER DYNAMICS (Paper Eq. 2)
# =============================================================================
np.random.seed(42)
q0, _, _, _ = get_stations_and_derivatives(0.0)

# Bounded initial perturbation around q_i(0)
angles = np.linspace(0, 2*np.pi, n, endpoint=False) + 0.2
radii = np.random.uniform(0.15, 0.22, size=n)
delta0_mat = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
x0 = (q0 + delta0_mat).flatten()

delta0_norm = np.linalg.norm(delta0_mat)
delta0_mean = np.mean(delta0_mat, axis=0, keepdims=True)
eta0_norm = np.linalg.norm(delta0_mat - delta0_mean)
rho_0 = np.sqrt(2.0) * eta0_norm
T_deadline = np.log(1.0 + delta0_norm**(1.0 - beta)) / (1.0 - beta)

def phi_beta(z):
    return np.sign(z) * (np.abs(z)**beta)

def closed_loop_rhs(t, x_flat):
    x = x_flat.reshape((n, d))
    q, dq, _, _ = get_stations_and_derivatives(t)
    delta = x - q

    diff_x = x[:, None, :] - x[None, :, :]
    dist_x = np.linalg.norm(diff_x, axis=-1)
    adj = ((dist_x < R_c) & (dist_x > 0)).astype(float)
    W = adj / (dist_x**nu + eps_w)
    L_w = np.diag(np.sum(W, axis=1)) - W

    coupling = (L_w @ delta) @ M_mat
    u = dq - delta - phi_beta(delta) - coupling
    return u.flatten()

# =============================================================================
# 5. SIMULATION EXECUTION & CERTIFICATE EVALUATION
# =============================================================================
print("Simulating Ring -> Chevron -> Ring maneuver...")
t_eval = np.linspace(0.0, t_final, 1401)
sol = solve_ivp(
    closed_loop_rhs, [0.0, t_final], x0,
    t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9
)

N_steps = len(t_eval)
x_traj = sol.y.T.reshape((N_steps, n, d))
q_traj = np.zeros_like(x_traj)
r_traj = np.zeros((N_steps, d))

delta_norm_t = np.zeros(N_steps)
eta_norm_t = np.zeros(N_steps)
R_bound_t = np.zeros(N_steps)
rho_H_t = np.zeros(N_steps)

d_q_t = np.zeros(N_steps)
l_H_t = np.zeros(N_steps)
s_margin_t = np.zeros(N_steps)
d_x_t = np.zeros(N_steps)
l_x_t = np.zeros(N_steps)

for k, t in enumerate(t_eval):
    q, _, r, _ = get_stations_and_derivatives(t)
    q_traj[k] = q
    r_traj[k] = r
    x = x_traj[k]

    delta = x - q
    eta = delta - np.mean(delta, axis=0, keepdims=True)
    delta_norm_t[k] = np.linalg.norm(delta)
    eta_norm_t[k] = np.linalg.norm(eta)

    inside = (1.0 + delta0_norm**(1.0 - beta)) * np.exp(-(1.0 - beta) * t) - 1.0
    R_bound_t[k] = (max(0.0, inside))**(1.0 / (1.0 - beta))
    rho_H_t[k] = min(np.sqrt(2.0) * R_bound_t[k], rho_0 * np.exp(-gamma_H * t))

    dist_q = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    dist_x = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    i_upper, j_upper = np.triu_indices(n, k=1)
    d_q_t[k] = np.min(dist_q[i_upper, j_upper])
    d_x_t[k] = np.min(dist_x[i_upper, j_upper])

    l_H_t[k] = max(np.linalg.norm(q[i] - q[j]) for i, j in backbone_edges)
    l_x_t[k] = max(np.linalg.norm(x[i] - x[j]) for i, j in backbone_edges)

    s_margin_t[k] = min(d_q_t[k] - d_safe, R_c - l_H_t[k])

mask_T = (t_eval > 1e-4) & (t_eval < T_deadline)
gamma_req = np.max(np.log(rho_0 / s_margin_t[mask_T]) / t_eval[mask_T])

print("\n--- FORMAL CERTIFICATE SUMMARY ---")
print(f"Initial tracking error ||delta_0|| : {delta0_norm:.4f} m")
print(f"Initial disagreement tube rho_0    : {rho_0:.4f} m")
print(f"Initial geometric margin s(0)      : {s_margin_t[0]:.4f} m (Condition rho_0 < s(0): {rho_0 < s_margin_t[0]})")
print(f"Finite-Time Arrival Deadline T     : {T_deadline:.4f} s")
print(f"Backbone algebraic conn. lambda_2  : {lambda_2_H:.4f}")
print(f"Required decay rate gamma_req      : {gamma_req:.4f}")
print(f"Achieved decay rate gamma_H (m={m_gain}): {gamma_H:.4f} (Certified: {gamma_H > gamma_req})")
print(f"Min nominal station dist d_q(t)    : {np.min(d_q_t):.4f} m")
print(f"Min actual agent dist d_x(t)       : {np.min(d_x_t):.4f} m (> d_safe = {d_safe} m)")
print(f"Max nominal backbone edge l_H(t)   : {np.max(l_H_t):.4f} m (< R_c = {R_c} m)")
print(f"Max actual backbone edge l_x(t)    : {np.max(l_x_t):.4f} m (< R_c = {R_c} m)")

# =============================================================================
# 6. FIGURE 1: 1x3 SUBPLOTS AT 3 INSTANTS DURING THE MANEUVER
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10.5,
    'legend.fontsize': 8.5
})

snap_times = [1.2, 8.0, 16.2]
snap_titles = [
    r"(a) Approach in Ring ($t = 1.2\,$s)",
    r"(b) Traversal in Chevron ($t = 8.0\,$s)",
    r"(c) Exit & Ring Recovery ($t = 16.2\,$s)"
]

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), sharey=True)
agent_colors = plt.cm.tab10(np.linspace(0, 1, n))

for ax, t_snap, title in zip(axes, snap_times, snap_titles):
    idx = np.argmin(np.abs(t_eval - t_snap))
    x_now = x_traj[idx]
    q_now = q_traj[idx]
    r_now = r_traj[idx]

    # Draw Corridor Walls
    rect_top = patches.Rectangle(
        (x_corr_start, y_wall), x_corr_end - x_corr_start, wall_height,
        facecolor='#d95f02', alpha=0.22, edgecolor='#a63603', lw=1.5, hatch='///', zorder=2
    )
    rect_bot = patches.Rectangle(
        (x_corr_start, -y_wall - wall_height), x_corr_end - x_corr_start, wall_height,
        facecolor='#d95f02', alpha=0.22, edgecolor='#a63603', lw=1.5, hatch='///', zorder=2
    )
    ax.add_patch(rect_top)
    ax.add_patch(rect_bot)

    # Draw Trajectories
    for i in range(n):
        ax.plot(
            x_traj[:idx+1, i, 0], x_traj[:idx+1, i, 1],
            color=agent_colors[i], lw=0.9, alpha=0.45, zorder=1
        )

    ax.plot([0, v_ref * t_final], [0, 0], 'k--', lw=1.0, alpha=0.5, label='Reference $r(t)$', zorder=1)

    # Draw 10-Cycle Backbone Edges
    edge_lines = [[x_now[i], x_now[j]] for i, j in backbone_edges]
    lc = LineCollection(edge_lines, colors='#2b5c8f', linewidths=1.4, alpha=0.8, zorder=3, label='Backbone $\mathcal{H}$')
    ax.add_collection(lc)

    # Draw Stations q_i and Agent States x_i
    ax.scatter(
        q_now[:, 0], q_now[:, 1], s=55, facecolors='none', edgecolors='k',
        linewidths=1.1, zorder=4, label='Stations $q_i(t)$'
    )
    for i in range(n):
        ax.scatter(
            x_now[i, 0], x_now[i, 1], s=35, color=agent_colors[i],
            edgecolors='k', linewidths=0.5, zorder=5,
            label='Agents $x_i(t)$' if i == 0 else ""
        )
        # Label agent numbers 1 to 10
        ax.text(
            x_now[i, 0] + 0.35, x_now[i, 1] + 0.35, f"{i+1}",
            fontsize=8, fontweight='bold', color='#1f2d3d', zorder=6
        )

    ax.scatter(r_now[0], r_now[1], marker='+', s=75, color='k', linewidths=1.5, zorder=7, label='Centroid $c(t)$')

    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_xlabel('$x$ [m]')
    ax.set_xlim(-5.0, 45.0)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)

axes[0].set_ylabel('$y$ [m]')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('corridor_maneuver_1x3.pdf', bbox_inches='tight')
plt.savefig('corridor_maneuver_1x3.png', dpi=300, bbox_inches='tight')
print("Saved 1x3 subplot figure to 'corridor_maneuver_1x3.pdf' and '.png'.")

# =============================================================================
# 7. FIGURE 2: SAFETY & CONNECTIVITY CERTIFICATES
# =============================================================================
fig_cert, ax_c = plt.subplots(1, 3, figsize=(13.5, 3.4))

# (a) Minimum pairwise distance
ax_c[0].plot(t_eval, d_x_t, color='#1b9e77', lw=1.8, label='$d_x(t)$ (observed)')
ax_c[0].plot(t_eval, d_q_t - rho_H_t, 'k--', lw=1.3, label=r'$d_q(t) - \rho_{\mathcal{H}}(t)$ (cert. bound)')
ax_c[0].axhline(d_safe, color='#d95f02', ls=':', lw=1.4, label=r'$d_{\mathrm{safe}} = 1.0\,$m')
ax_c[0].set_title('(a) Min Pairwise Distance $d_x(t)$')
ax_c[0].set_xlabel('Time $t$ [s]')
ax_c[0].set_ylabel('Distance [m]')
ax_c[0].set_ylim(0.5, 3.0)
ax_c[0].legend(loc='lower right')
ax_c[0].grid(True, ls=':', alpha=0.4)

# (b) Maximum backbone edge length
ax_c[1].plot(t_eval, l_x_t, color='#1b9e77', lw=1.8, label=r'$\ell_x(t)$ (observed)')
ax_c[1].plot(t_eval, l_H_t + rho_H_t, 'k--', lw=1.3, label=r'$\ell_{\mathcal{H}}(t) + \rho_{\mathcal{H}}(t)$ (cert. bound)')
ax_c[1].axhline(R_c, color='#d95f02', ls=':', lw=1.4, label=r'$R_c = 6.0\,$m')
ax_c[1].set_title(r'(b) Max Backbone Edge $\ell_x(t)$')
ax_c[1].set_xlabel('Time $t$ [s]')
ax_c[1].set_ylabel('Length [m]')
ax_c[1].set_ylim(1.0, 6.5)
ax_c[1].legend(loc='upper right')
ax_c[1].grid(True, ls=':', alpha=0.4)

# (c) Geometric margin s(t) vs Disagreement Tube rho_H(t)
ax_c[2].plot(t_eval, s_margin_t, 'k-', lw=1.6, label='Geometric margin $s(t)$')
ax_c[2].plot(t_eval, rho_H_t, color='#1b9e77', ls='--', lw=1.6, label=r'Disagreement tube $\rho_{\mathcal{H}}(t)$')
ax_c[2].fill_between(t_eval, rho_H_t, s_margin_t, color='#1b9e77', alpha=0.18, label=r'Certified margin ($s > \rho_{\mathcal{H}}$)')
ax_c[2].axvline(T_deadline, color='#7570b3', ls='-.', lw=1.2, label=f'Deadline $T = {T_deadline:.2f}\,$s')
ax_c[2].set_title(r'(c) Margin $s(t)$ vs. Tube $\rho_{\mathcal{H}}(t)$')
ax_c[2].set_xlabel('Time $t$ [s]')
ax_c[2].set_ylabel('Margin [m]')
ax_c[2].set_xlim(0, 4.0)
ax_c[2].set_ylim(0, 1.8)
ax_c[2].legend(loc='upper right')
ax_c[2].grid(True, ls=':', alpha=0.4)

plt.tight_layout()
plt.savefig('corridor_certificates.pdf', bbox_inches='tight')
plt.savefig('corridor_certificates.png', dpi=300, bbox_inches='tight')
print("Saved certificate verification plots to 'corridor_certificates.pdf' and '.png'.")

# =============================================================================
# 8. ANIMATION GENERATION
# =============================================================================
fig_anim, ax_a = plt.subplots(figsize=(11, 4.2))
rect_top_a = patches.Rectangle(
    (x_corr_start, y_wall), x_corr_end - x_corr_start, wall_height,
    facecolor='#d95f02', alpha=0.25, edgecolor='#a63603', lw=1.5, hatch='///'
)
rect_bot_a = patches.Rectangle(
    (x_corr_start, -y_wall - wall_height), x_corr_end - x_corr_start, wall_height,
    facecolor='#d95f02', alpha=0.25, edgecolor='#a63603', lw=1.5, hatch='///'
)
ax_a.add_patch(rect_top_a)
ax_a.add_patch(rect_bot_a)
ax_a.plot([0, v_ref * t_final], [0, 0], 'k--', lw=1.0, alpha=0.4)

ax_a.set_xlim(-5.0, 45.0)
ax_a.set_ylim(-5.5, 5.5)
ax_a.set_aspect('equal')
ax_a.set_xlabel('$x$ [m]')
ax_a.set_ylabel('$y$ [m]')
ax_a.grid(True, ls=':', alpha=0.4)

x_init = x_traj[0]
q_init = q_traj[0]
edge_init = [[x_init[i], x_init[j]] for i, j in backbone_edges]

lc_anim = LineCollection(edge_init, colors='#2b5c8f', linewidths=1.4, alpha=0.8, zorder=3)
ax_a.add_collection(lc_anim)
stations_scatter = ax_a.scatter(q_init[:, 0], q_init[:, 1], s=60, facecolors='none', edgecolors='k', lw=1.1, zorder=4)
agents_scatter = ax_a.scatter(x_init[:, 0], x_init[:, 1], s=40, c=agent_colors, edgecolors='k', lw=0.5, zorder=5)
agent_labels = [
    ax_a.text(x_init[i, 0] + 0.35, x_init[i, 1] + 0.35, f"{i+1}", fontsize=8, fontweight='bold', zorder=6)
    for i in range(n)
]
trails = [ax_a.plot([], [], color=agent_colors[i], lw=1.0, alpha=0.4)[0] for i in range(n)]
title_text = ax_a.set_title('', fontweight='bold')

frame_step = 8
frame_indices = np.arange(0, N_steps, frame_step)

def update_anim(frame_idx):
    x_now = x_traj[frame_idx]
    q_now = q_traj[frame_idx]
    t_now = t_eval[frame_idx]

    stations_scatter.set_offsets(q_now)
    agents_scatter.set_offsets(x_now)
    lc_anim.set_segments([[x_now[i], x_now[j]] for i, j in backbone_edges])

    for i in range(n):
        agent_labels[i].set_position((x_now[i, 0] + 0.35, x_now[i, 1] + 0.35))

    trail_start = max(0, frame_idx - 160)
    for i in range(n):
        trails[i].set_data(x_traj[trail_start:frame_idx+1, i, 0], x_traj[trail_start:frame_idx+1, i, 1])

    title_text.set_text(
        f"Corridor Maneuver (Ring -> Chevron -> Ring) | t = {t_now:5.2f} s | "
        f"min d_x = {d_x_t[frame_idx]:.2f} m | max l_x = {l_x_t[frame_idx]:.2f} m"
    )
    return [lc_anim, stations_scatter, agents_scatter, title_text] + agent_labels + trails

ani = animation.FuncAnimation(fig_anim, update_anim, frames=frame_indices, interval=40, blit=True)
ani.save('corridor_maneuver.gif', writer='pillow', fps=25)
print("Saved animation to 'corridor_maneuver.gif'.")