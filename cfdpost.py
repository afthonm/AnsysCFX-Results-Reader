# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 05:31:29 2021

@author: ASUS
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt

class Atmosphere:
    def __init__(self, h, dT=0.):
        self.h = h
        self.dT = dT
        self.calculate()
        
    def calculate(self):
        p0, T0, mu0 = 101325, 288.15, 1.81206e-5
        g, M, R, L = 9.80665, 0.0289644, 8.3144598, -0.0065
        
        fp = ((T0 + L*self.h)/T0)**(-g*M/R/L)    
        T = T0 + L*self.h + self.dT
        p = fp * p0
        rho = p/(R/M)/T
        mu = mu0 * (0.555*1.8*T0+120)/(0.555*1.8*T+120) * (T/T0)**1.5
        a = (1.4*R/M*T)**0.5
        
        self.T, self.p, self.rho, self.mu, self.a = T, p, rho, mu, a

class Reference:
    def __init__(self, Lref, Sref, reftype=1, **kwargs):
        self.L = Lref
        self.S = Sref
        
        if reftype==1:
            self.reftype1(**kwargs)
        elif reftype==2:
            self.reftype2(**kwargs)
        
        self.q = 0.5*self.rho*self.V**2
        self.qS = self.q*Sref
        
    def reftype1(self, Re, Ma, T=288.15, mu=1.8e-5):
        a = (1.4*287*T)**0.5
        V = Ma * a
        rho = mu*Re/V/self.L
        p = rho*287*T
        
        self.Re, self.Ma, self.T, self.mu = Re, Ma, T, mu
        self.a, self.V, self.rho, self.p = a, V, rho, p
        
    def reftype2(self, Ma, Alt, ISA=0.):
        atm = Atmosphere(Alt, ISA)
        
        V = Ma*atm.a
        Re = atm.rho*V*self.L/atm.mu
        
        self.Re, self.Ma, self.T, self.mu = Re, Ma, atm.T, atm.mu
        self.a, self.V, self.rho, self.p = atm.a, V, atm.rho, atm.p

class CaseAranger:
    def __init__(self, name, files, partidx='#', delimiter='\t', save_to=None,
                 insert_columns=None):
        self.name = name
        self.files = files
        self.partidx = partidx
        self.delimiter = delimiter
        self.insertcols = insert_columns
        
        self.arange()
        if save_to is not None:
            for part,df in self.parts.items():
                df.to_csv(save_to+'%s_%s.txt'%(self.name,part), sep=delimiter,
                          index=False)
        
    def arange(self):
        parts = {}
        for i,file in enumerate(self.files):
            with open(file) as f:
                lines = f.readlines()
                for j,line in enumerate(lines):
                    line = line.split(self.delimiter)
                    x = len(self.partidx)
                    if line[0][:x] == self.partidx:
                        name = line[0][x:]
                        values = lines[j+2].split(self.delimiter)
                        values = [float(v) for v in values]
                        if i == 0:
                            column = lines[j+1].split(self.delimiter)
                            column[-1] = column[-1][:-1]
                            parts[name] = pd.DataFrame([values], columns=column)
                        else:
                            parts[name].loc[len(parts[name].index)] = values
        for name,df in parts.items():
            if self.insertcols is not None:
                df.insert(**self.insertcols)
        self.parts = parts

class PlotIt:
    def __init__(self, name, results, resnames, styles=None, save_to=None, featured=False):
        self.name = name
        self.results = results
        self.resnames = resnames
        self.styles = styles
        self.save_to = save_to
        self.visualize(featured)
    
    def visualize(self, featured=False):
        plt.close('all')
        fig1, ax1 = plt.subplots(1,2,figsize=[14,7])
        fig2, ax2 = plt.subplots(1,3,figsize=[14,7], sharey=True)
        fig3, ax3 = plt.subplots(1,3,figsize=[14,7])
        fig4, ax4 = plt.subplots(1,2,figsize=[14,7])
        
        maxCL = max([max(res.CL) for res in self.results])
        minCL = min([min(res.CL) for res in self.results])
        xlabels = [['Angle of Attack [deg]']*2, ['Angle of Attack [deg]']*3,
                   ['Angle of Attack [deg]']*3,
                   ['$C_L$ [-]','Angle of Attack [deg]']]
        if featured:
            ylabels = [['$C_L$ [-]', '$C_D$ [dc]'],
                       ['$C_M$ [-]', None, None],
                       ['$C_Y$ x $10^{-3}$ [-]', '$C_{Roll}$ x $10^{-3}$ [-]',
                        '$C_{Yaw}$ x $10^{-3}$ [-]'],
                       ['$C_D$ [-]','$C_L/C_D$ [-]']]
            kc, kd = 1e3, 1e4
            axisld = None
            axislat = None
        else:
            ylabels = [['$C_L$ [-]', '$C_D$ [-]'], ['$C_M$ [-]', None, None],
                       ['$C_Y$ [-]', '$C_{Roll}$ [-]', '$C_{Yaw}$ [-]'],
                       ['$C_D$ [-]','$C_L/C_D$ [-]']]
            kc, kd, kk = 1, 1, 0.05
            axisld = [None,None,minCL-kk,maxCL+kk]
            axislat = [None,None,minCL-kk,maxCL+kk]
            
        suptitles = ['Lift & Drag Coefficient %s'%self.name,
                     'Longitudinal Moment Coefficient %s'%self.name,
                     'Lateral-Directional Force & Moment Coefficient %s'%self.name,
                     'Drag Polar & Aerodynamic Efficiency %s'%self.name]
        titles = [[None]*2, ['at CG', 'at CG-20%', 'at CG+20%'], [None]*3, [None]*2]
        axislim = [[axisld,None], [axisld]*3, [axislat]*3, [None]*2]
        
        if self.styles is None:
            self.styles = [None for _ in range(len(self.results))]
        for i, res in enumerate(self.results):
            label = self.resnames[i]
            style = self.styles[i]
            if style is None:
                ax1[0].plot(res.alpha, res.CL, label=label)
                ax1[1].plot(res.alpha, res.CD*kd, label=label)
                
                ax2[0].plot(res.alpha, res.CM, label=label)
                try:
                    ax2[1].plot(res.alpha, res.CM_fcg, label=label)
                    ax2[2].plot(res.alpha, res.CM_acg, label=label)
                except:
                    pass
                
                ax3[0].plot(res.alpha, res.CY*kc, label=label)
                ax3[1].plot(res.alpha, res.CR*kc, label=label)
                ax3[2].plot(res.alpha, res.CN*kc, label=label)
                
                ax4[0].plot(res.CL, res.CD, label=label)
                ax4[1].plot(res.alpha, res.CL/res.CD, label=label)
            else:
                ax1[0].plot(res.alpha, res.CL, style, label=label)
                ax1[1].plot(res.alpha, res.CD*kc, style, label=label)
                
                ax2[0].plot(res.alpha, res.CM, style, label=label)
                try:
                    ax2[1].plot(res.alpha, res.CM_fcg, style, label=label)
                    ax2[2].plot(res.alpha, res.CM_acg, style, label=label)
                except:
                    pass
        
                ax3[0].plot(res.alpha, res.CY*kc, style, label=label)
                ax3[1].plot(res.alpha, res.CR*kc, style, label=label)
                ax3[2].plot(res.alpha, res.CN*kc, style, label=label)
                
                ax4[0].plot(res.CL, res.CD, style, label=label)
                ax4[1].plot(res.alpha, res.CL/res.CD, style, label=label)
            
        figs = [fig1, fig2, fig3, fig4]
        for i,ax in enumerate([ax1, ax2, ax3, ax4]):
            for j,x in enumerate(ax):
                x.set_xlabel(xlabels[i][j])
                if ylabels[i][j] is not None:
                    x.set_ylabel(ylabels[i][j])
                x.grid()
                x.grid(which='minor', linestyle=':')
                x.minorticks_on()
                x.legend()
                if axislim[i][j] is not None:
                    x.axis(axislim[i][j])
                if titles[i][j] is not None:
                    x.set_title(titles[i][j], loc='left', fontsize=10)
            figs[i].suptitle(suptitles[i])
            figs[i].tight_layout()
            figs[i].subplots_adjust(top=0.9)
            if self.save_to is not None:
                figs[i].savefig(self.save_to+suptitles[i])
            else:
                figs[i].show()
        self.figures = figs

class Case:
    def __init__(self, name, Reference):
        self.name = name
        self.ref = Reference
        
    def input_resdata(self, resfiles, partnames, delimiter='\t'):
        self.data = {}
        for part, file in tuple(zip(partnames,resfiles)):
            self.data[part] = pd.read_csv(file, delimiter=delimiter)
            try:
                self.data[part].sort_values('alpha')
            except:
                pass
    
    def input_dataframe(self, dataframes, partnames=[]):
        self.data = {}
        for part, df in list(zip(partnames,dataframes)):
            self.data[part] = df
            try:
                self.data[part].sort_values('alpha')
            except:
                pass
            
    def analyze(self):
        results = {}
        for part, data in self.data.items():
            alpha = data.alpha.to_numpy()
            D = data.Fx * np.cos(np.deg2rad(alpha)) +\
                data.Fz * np.sin(np.deg2rad(alpha))
            L = -data.Fx * np.sin(np.deg2rad(alpha)) +\
                data.Fz * np.cos(np.deg2rad(alpha))
            
            CD = D.to_numpy()/self.ref.qS
            CL = L.to_numpy()/self.ref.qS
            CY = data.Fy.to_numpy()/self.ref.qS
            
            CM = data.My.to_numpy()/self.ref.qS/self.ref.L
            CMa = np.diff(CM)/np.diff(alpha)
            CMa = np.hstack((CMa,CMa[-1]))
            
            dCM_dCL = np.diff(CM)/np.diff(CL)
            dCM_dCL = np.hstack((dCM_dCL,dCM_dCL[-1]))
            
            CR = data.Mx.to_numpy()/self.ref.qS/self.ref.L
            CN = data.Mz.to_numpy()/self.ref.qS/self.ref.L
            
            res = {'CD': CD, 'CL': CL, 'CY': CY, 'CM': CM, 'CR': CR, 'CN': CN,
                   'CMa': CMa, 'dCM_dCL': dCM_dCL}
            
            try:
                CM_fcg = data.My_fcg.to_numpy()/self.ref.qS/self.ref.L
                CMa_fcg = np.diff(CM_fcg)/np.diff(alpha)
                CMa_fcg = np.hstack((CMa_fcg,CMa_fcg[-1]))
                res.update({'CM_fcg': CM_fcg, 'CMa_fcg': CMa_fcg})
            except:
                pass
                
            try:
                CM_acg = data.My_acg.to_numpy()/self.ref.qS/self.ref.L
                CMa_acg = np.diff(CM_acg)/np.diff(alpha)
                CMa_acg = np.hstack((CMa_acg,CMa_acg[-1]))
                res.update({'CM_acg': CM_acg, 'CMa_acg': CMa_acg})
            except:
                pass
            
            results[part] = pd.DataFrame(res)
        
        for i, part in enumerate(self.data.keys()):
            if i == 0:
                X = results[part].to_numpy()
            else:
                X = X + results[part].to_numpy()
        X = pd.DataFrame(X, columns=results[part].columns)
        results['Total'] = X
        
        alpha = self.data[part].alpha
        for part in results.keys():
            results[part].insert(0, column='alpha', value=alpha)
        self.results = results
        
    def visualize(self, styles=None, save_to=None, featured=False):
        self.plots = PlotIt(self.name, self.results.values(),
                            list(self.results.keys()), styles, save_to,
                            featured)
        
    def print_data(self, save_to='', filetype='txt', delimiter='\t'):
        for part, df in self.results.items():
            file = save_to + '%s %s.%s'%(self.name,part,filetype)
            df.to_csv(file, sep=delimiter, index=False, float_format='%.4f')
        
class Analyses:
    def __init__(self, name, Cases):
        self.name = name
        self.Cases = Cases
        
    def analyze(self):
        for case in self.Cases:
            case.analyze()
    
    def visualize(self, styles=None, save_to=None, featured=False):
        results = [case.results['Total'] for case in self.Cases]
        resnames = [case.name for case in self.Cases]
        self.plots = PlotIt(self.name, results, resnames, styles, save_to, featured)