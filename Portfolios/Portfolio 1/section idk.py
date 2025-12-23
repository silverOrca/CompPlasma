# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 15:02:21 2025

@author: ciara


"""


        # create a small inset to show a zoomed section of the kinetic-energy curve
        # position the inset slightly offset per particle to reduce overlap
        inset_x = 0.62
        inset_y = 0.62 - i * 0.13
        inset_w = 0.33
        inset_h = 0.24
        axins = ek_axis.inset_axes([inset_x, inset_y, inset_w, inset_h])
        axins.plot(time, e_k, color=colors[i], linewidth=0.9)

        # center the zoom on the kinetic-energy peak and use a small time window
        idx_peak = int(e_k.argmax())
        t_center = float(time[idx_peak])
        t_total = float(time[-1] - time[0])
        zoom_width = max(t_total * 0.06, t_total * 0.02)
        t0 = max(time[0], t_center - zoom_width / 2)
        t1 = min(time[-1], t_center + zoom_width / 2)
        axins.set_xlim(t0, t1)

        # set y-limits for the inset from the data in the zoom window with padding
        mask = (time >= t0) & (time <= t1)
        if mask.any():
            e_min = float(e_k[mask].min())
            e_max = float(e_k[mask].max())
            ypad = (e_max - e_min) * 0.2 if (e_max - e_min) > 0 else e_max * 0.1 + 1e-9
            axins.set_ylim(e_min - ypad, e_max + ypad)

        axins.tick_params(axis='both', which='major', labelsize=6)
        axins.set_title('zoom', fontsize=7)