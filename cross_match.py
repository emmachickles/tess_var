# ==============================================================================
# ==============================================================================
# 
# This script contains functions for cross matching mergen classifications with
# stellar variabiilty catalogs (e.g. ASAS-SN, Simbad, GCVS)
# 
# // Emma Chickles
# 
# -- existing functions --------------------------------------------------------
# 
# querying catalogs:
# * query_simbad 
# * query_gcvs
# * query_asas_sn
# 
# organizing object types:
# 
# 
# -- TODO ----------------------------------------------------------------------
# 
# * deal with dependencies on data_dir (clarify what is required to run these
#   functions)
#   * should there be a function that sets up the folder structure? or should
#   * that happen in mergen?
# * currently only handles the 2 minute cadence TICIDs (fast cadence during 
#   Cycles 1 and 2)
# 
# ==============================================================================
# ==============================================================================

from __init__ import *

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Query catalogs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def query_simbad(sector='all', data_dir='data/', query_mast=False):
    '''Cross-matches ASAS-SN catalog with TIC catalog based on matching GAIA IDs
    * data_dir
    * sector: 'all' or int, currently only handles short-cadence'''

    # import time
    
    customSimbad = Simbad()
    customSimbad.add_votable_fields('otypes')
    # customSimbad.add_votable_fields('biblio')

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        print('Sector '+str(sector))
        out_f = data_dir+'databases/Sector'+str(sector)+'_simbad.txt'

        ticid_simbad = []
        otypes_simbad = []
        main_id_simbad = []
        bibcode_simbad = []

        with open(out_f, 'a') as f: # >> make file if not already there
            f.write('')    

        with open(out_f, 'r') as f:
            lines = f.readlines()
            ticid_already_classified = []
            for line in lines:
                ticid_already_classified.append(float(line.split(',')[0]))

        if not query_mast:
            tic_cat=pd.read_csv(data_dir+'Sector'+str(sector)+'/Sector'+str(sector)+\
                                     'tic_cat_all.csv', index_col=False)
            ticid_list = tic_cat['ID']

        print(str(len(ticid_list))+' targets')
        print(str(len(ticid_already_classified))+' targets completed')
        ticid_list = np.setdiff1d(ticid_list, ticid_already_classified)
        print(str(len(ticid_list))+' targets to query')

        count = 0
        for tic in ticid_list:

            count += 1
            res = None

            while res is None:
                try:
                    print(str(count)+'/'+str(len(ticid_list))+\
                          ': finding object type for Sector ' +str(sector)+\
                          ' TIC' + str(int(tic)))

                    target = 'TIC ' + str(int(tic))                    
                    if query_mast:
                        # >> get coordinates
                        catalog_data = Catalogs.query_object(target, radius=0.02,
                                                             catalog='TIC')[0]

                    else:
                        ind = np.nonzero(tic_cat['ID'].to_numpy() == tic)[0][0]
                        catalog_data=tic_cat.iloc[ind]
                    # time.sleep(6)


                    # -- get object type from Simbad ---------------------------

                    # >> first just try querying Simbad with the TICID
                    res = customSimbad.query_object(target)
                    # time.sleep(6)

                    # >> if no luck with that, try checking other catalogs
                    catalog_names = ['TYC', 'HIP', 'TWOMASS', 'SDSS', 'ALLWISE',
                                     'GAIA', 'APASS', 'KIC']
                    for name in catalog_names:
                        if type(res) == type(None):
                            if type(catalog_data[name]) != np.ma.core.MaskedConstant:
                                target_new = name + ' ' + str(catalog_data[name])
                                res = customSimbad.query_object(target_new)
                                # time.sleep(6)


                    if type(res) == type(None):
                        print('failed :(')
                        res=0 
                        with open(out_f, 'a') as f:
                            f.write('{},{},{}\n'.format(tic, '', ''))              
                        ticid_simbad.append(tic)
                        otypes_simbad.append('none')
                        main_id_simbad.append('none')                
                    else:
                        otypes = res['OTYPES'][0].decode('utf-8')
                        main_id = res['MAIN_ID'].data[0].decode('utf-8')
                        ticid_simbad.append(tic)
                        otypes_simbad.append(otypes)
                        main_id_simbad.append(main_id)

                        with open(out_f, 'a') as f:
                            f.write('{},{},{}\n'.format(tic, otypes, main_id))

                    # time.sleep(6)
                except:
                    pass
                    print('connection failed! Trying again now')
            
def query_gcvs(data_dir='./', sector='all', tol=0.1, diag_plot=True):
    '''Cross-matches GCVS catalog with TIC catalog.
    * data_dir
    * sector: 'all' or int, currently only handles short-cadence
    * tol: maximum separation of TIC target and GCVS target (in arcsec)
    '''
    data = pd.read_csv(data_dir+'gcvs_database.csv')
    print('Loaded gcvs_database.csv')
    data_coords = coord.SkyCoord(data['RAJ2000'], data['DEJ2000'],
                                 unit=(u.hourangle, u.deg))

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        prefix = data_dir+'databases/Sector'+str(sector)+'_gcvs'
        out_fname = prefix+'.txt'

        sector_data = pd.read_csv(data_dir+'Sector'+str(sector)+\
                                  '/Sector'+str(sector)+'tic_cat_all.csv',
                                  index_col=False)
        print('Loaded Sector'+str(sector)+'tic_cat_all.csv')

        # >> find GCVS target closest to each TIC target
        if os.path.exists(prefix+'_sep.txt'):
            sep_arcsec = np.loadtxt(prefix+'_sep.txt')
            min_inds = np.loadtxt(prefix+'_sep_inds.txt').astype('int')
            print('Loaded '+prefix+'_sep.txt')
        else:
            min_sep = []
            min_inds = []
            for i in range(len(sector_data)):
                print('TIC '+str(int(sector_data['ID'][i]))+'\t'+str(i)+'/'+\
                      str(len(sector_data)))
                ticid_coord = coord.SkyCoord(sector_data['ra'][i],
                                             sector_data['dec'][i],
                                             unit=(u.deg, u.deg)) 
                sep = ticid_coord.separation(data_coords)
                min_sep.append(np.nanmin(sep))
                ind = np.nanargmin(sep)
                min_inds.append(ind)

            sep_arcsec = np.array([sep.to(u.arcsec).value for sep in min_sep])
            min_inds = np.array(min_inds).astype('int')
            np.savetxt(prefix+'_sep.txt', sep_arcsec)
            np.savetxt(prefix+'_sep_inds.txt', min_inds)

        # >> save the variability type if GCVS target is close enough
        with open(out_fname, 'w') as f:

            for i in range(len(sector_data)):
                if sep_arcsec[i] < tol:
                    ind = min_inds[i]            
                    f.write(str(int(sector_data['ID'][i]))+','+\
                            str(data['VarType'][ind])+','+\
                            str(data['VarName'][ind])+'\n')

                else:
                    f.write(str(int(sector_data['ID'][i]))+',,\n')

        # >> plotting
        if diag_plot:
            # >> make histogram of minimum separations
            fig, ax = plt.subplots()
            bins = 10**np.linspace(np.floor(np.log10(np.nanmin(sep_arcsec))),
                                   np.ceil(np.log10(np.nanmax(sep_arcsec))), 50)
            ax.hist(sep_arcsec, bins=bins, log=True)
            ax.set_xlabel('arcseconds')
            ax.set_ylabel('number of targets in Sector '+str(sector))
            ax.set_xscale('log')
            fig.savefig(prefix+'_sep_arcsec.png')
            
            # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
            tol_tests = [10, 1, 0.1] 
            for tol in tol_tests:
                inds1 = np.nonzero(sep_arcsec < tol)
                inds2 = min_inds[inds1]
                print('Tolerance: '+str(tol)+' arcseconds, number of targets: '+\
                      str(len(inds1[0])))
                plt.figure()
                plt.plot(sector_data['GAIAmag'][inds1[0]], data['magMax'][inds2], '.k')
                plt.xlabel('GAIA magnitude (TIC)')
                plt.ylabel('magMax (GCVS)')
                plt.savefig(prefix+'_tol'+str(tol)+'.png')
                plt.close()
                
def query_asas_sn(data_dir='./', sector='all', diag_plot=True):
    '''Cross-matches ASAS-SN catalog with TIC catalog based on matching GAIA IDs
    * data_dir
    * sector: 'all' or int, currently only handles short-cadence
    '''
    data = pd.read_csv(data_dir+'asas_sn_database.csv')
    print('Loaded asas_sn_database.csv')
    data_coords = coord.SkyCoord(data['RAJ2000'], data['DEJ2000'],
                                 unit=(u.deg, u.deg))

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        # >> could also have retrieved ra dec from all_targets_S*_v1.txt
        sector_data = pd.read_csv(data_dir+'Sector'+str(sector)+\
                                  '/Sector'+str(sector)+'tic_cat_all.csv',
                                  index_col=False)
        print('Loaded Sector'+str(sector)+'tic_cat_all.csv')
        out_fname = data_dir+'databases/Sector'+str(sector)+'_asassn.txt'

        _, comm1, comm2 = np.intersect1d(sector_data['GAIA'], data['GDR2_ID'],
                                         return_indices=True)

        # >> save cross-matched target in text file
        with open(out_fname, 'w') as f:
            for i in range(len(sector_data)):            
                if i in comm1:
                    ind = comm2[np.nonzero(comm1 == i)][0]
                    f.write(str(int(sector_data['ID'][i]))+','+\
                            str(data['Type'][ind])+','+str(data['ID'][ind])+'\n')
                else:
                    f.write(str(int(sector_data['ID'][i]))+',,\n')
        print('Saved '+out_fname)

        if diag_plot:
            prefix = data_dir+'databases/Sector'+str(sector)+'_'

            # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
            plt.figure()
            plt.plot(sector_data['GAIAmag'][comm1], data['Mean Vmag'][comm2], '.k')
            plt.xlabel('GAIA magnitude (TIC)')
            plt.ylabel('Mean Vmag (ASAS-SN)')
            plt.savefig(prefix+'asassn_mag_cross_match.png')
            plt.close()

            # >> get minimum separations between TIC and ASAS-SN targts
            if os.path.exists(prefix+'asassn_sep.txt'):
                sep_arcsec = np.loadtxt(prefix+'asassn_sep.txt')
                min_inds = np.loadtxt(prefix+'asassn_sep_inds.txt').astype('int')
            else:
                min_sep = []
                min_inds = []
                for i in range(len(sector_data)):
                    print('TIC '+str(int(sector_data['ID'][i]))+'\t'+str(i)+'/'+\
                          str(len(sector_data)))
                    ticid_coord = coord.SkyCoord(sector_data['ra'][i],
                                                 sector_data['dec'][i],
                                                 unit=(u.deg, u.deg)) 
                    sep = ticid_coord.separation(data_coords)
                    min_sep.append(np.min(sep))
                    min_inds.append(np.argmin(sep))
                sep_arcsec = np.array([sep.to(u.arcsec).value for sep in min_sep])
                min_inds = np.array(min_inds)
                np.savetxt(prefix+'asassn_sep.txt', sep_arcsec)
                np.savetxt(prefix+'asassn_sep_inds.txt', min_inds)

            # >> make histogram of minimum separations
            fig, ax = plt.subplots()
            ax.hist(sep_arcsec, bins=10**np.linspace(-2, 4, 30), log=True)
            ax.set_xlabel('arcseconds')
            ax.set_ylabel('number of targets in Sector '+str(sector))
            ax.set_xscale('log')
            fig.savefig(prefix+'asassn_sep_arcsec.png')

            fig1, ax1 = plt.subplots()
            ax1.hist(sep_arcsec[comm1], bins=10**np.linspace(-2, 4, 30), log=True)
            ax1.set_xlabel('arcseconds')
            ax1.set_ylabel('number of cross-matched targets in Sector '+str(sector))
            ax1.set_xscale('log')
            ax1.set_ylim(ax.get_ylim())
            fig1.savefig(prefix+'asassn_sep_cross_match.png')

            # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
            tol_tests = [10, 1, 0.1]
            for tol in tol_tests:
                inds1 = np.nonzero(sep_arcsec < tol)
                inds2 = min_inds[inds1]
                plt.figure()
                plt.plot(sector_data['GAIAmag'][inds1][0],
                         data['Mean Vmag'][inds2], '.k')
                plt.xlabel('GAIA magnitude (TIC)')
                plt.ylabel('Mean Vmag (ASAS-SN)')
                plt.savefig(prefix+'asassn_mag_tol'+str(tol)+'.png')
                plt.close()


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Organizing object types :::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
def get_otype_descr():
    
    d = {}
    with open('docs/otype_descr.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            key = line.split(',')[0]
            val = ','.join(line.split(',')[1:])[:-2]
            d[key] = val

    return d
        

def get_otype_dict(data_dir='/nfs/blender/data/tdaylan/data/',
                   uncertainty_flags=[':', '?', '*']):
    '''Return a dictionary of object type descriptions.'''
    
    d = {}
        
    with open(data_dir + 'gcvs_labels.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        otype, description = line.split(' = ')
        
        # >> remove uncertainty flags
        if otype[-1] in uncertainty_flags:
            otype = otype[:-1]
        
        # >> remove new line character
        description = description.replace('\n', '')
        
        d[otype] = description
        
    return d


def make_parent_dict():    
    d = {'I': ['IA', 'IB'],
         'IN': ['FU', 'INA', 'INB', 'INTIT', 'IN(YY)', 'INAT', 'INS', 'INSA',
                'INSB', 'INST', 'INT', 'INT(YY)', 'IT'],
         'IS': ['ISA', 'ISB'],
         'Fl': ['UV', 'UVN'],
         
         'BCEP': ['BCEPS', 'BCEP(B)'],
         'CEP': ['CEP(B)', 'DCEP', 'DCEPS'],
         'CW': ['CWA', 'CWB'],
         'DSCT': ['DSCTC', 'DSCTC(B)'],
         'L': ['LB', 'LC', 'LPB', 'LP', 'LBV'],
         'RR': ['RR(B)', 'RRAB', 'RRC'],
         'RV': ['RVA', 'RVB'],
         'SR': ['SRA', 'SRB', 'SRC', 'SRD', 'SRS'],
         'ZZ': ['ZZA', 'ZZB', 'ZZO', 'ZZLep'],
         
         'ACV': ['ACVO'],
         
         'N': ['NA', 'NB', 'NC', 'NL', 'NR'],
         'SN': ['SNI', 'SNII'],
         'UG': ['UGSS', 'UGSU', 'UGZ'],
         
         # 'E': ['EA', 'EB', 'EP', 'EW'],
         'D': ['DM', 'DS', 'DW'],
         'K': ['KE', 'KW'],
         
         'X': ['XB', 'XF', 'XI', 'XJ', 'XND', 'XNG', 'XP', 'XPR',
               'XPRM', 'XM', 'XRM', 'XN','XNA','XNGP','XPM','XPNG',
               'XNP'],
         }    

    parents = list(d.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(d[parent])

    return d, parents, subclasses

def make_variability_tree():
    var_d = {'eruptive':
         ['Fl', 'BE', 'FU', 'GCAS', 'I', 'IA', 'IB', 'IN', 'INA', 'INB', 'INT,IT',
          'IN(YY)', 'IS', 'ISA', 'ISB', 'RCB', 'RS', 'SDOR', 'UV', 'UV', 'UVN',
          'WR', 'INTIT', 'GCAS'],
         'pulsating':
             ['Pu', 'ACYG', 'BCEP', 'BCEPS', 'BLBOO', 'CEP', 'CEP(B)', 'CW', 'CWA',
              'CWB', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'GDOR', 'L', 'LB', 'LC',
              'LPB', 'M', 'PVTEL', 'RPHS', 'RR', 'RR(B)', 'RRAB', 'RRC', 'RV',
              'RVA', 'RVB', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
              'ZZ', 'ZZA', 'ZZB', 'ZZO'],
         'rotating': ['ACV', 'ACVO', 'BY', 'ELL', 'FKCOM', 'PSR',
                      'R', 'SXARI'],
         'cataclysmic':
             ['N', 'NA', 'NB', 'NC', 'NL', 'NR', 'SN', 'SNI', 'SNII', 'UG',
              'UGSS', 'UGSU', 'UGZ', 'ZAND'],
         'eclipsing':
             ['E', 'EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR', 'AR',
              'D', 'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD'],
             'xray':
             ['AM', 'X', 'XB', 'XF', 'XI', 'XJ', 'XND', 'XNG', 'XP', 'XPR', 'XPRM',
              'XM'],
             'other': ['VAR']} 
    return var_d

def make_redundant_otype_dict():
    # >> keys are redundant object types, and will be removed if star is also
    # >> classified as any of the associated dictionary values
    var_d = make_variability_tree()

    d = {'**': var_d['eclipsing']+['R'],
         'E':  ['EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR', 'AR', 'D',
                'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD'],
         'Er': var_d['eruptive'],
         'ROT': var_d['rotating'],
         'Ro': var_d['rotating'],
         'Pu': var_d['pulsating'],
         'L': var_d['eruptive']+var_d['rotating']+\
         var_d['cataclysmic']+var_d['eclipsing']+var_d['xray']+var_d['other']+\
        ['Pu', 'ACYG', 'BCEP', 'BCEPS', 'BLBOO', 'CEP', 'CEP(B)', 'CW', 'CWA',
              'CWB', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'GDOR', 'LB', 'LC',
              'LPB', 'M', 'PVTEL', 'RPHS', 'RR', 'RR(B)', 'RRAB', 'RRC', 'RV',
              'RVA', 'RVB', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
              'ZZ', 'ZZA', 'ZZB', 'ZZO'],
         'LP': var_d['eruptive']+var_d['pulsating']+var_d['rotating']+\
         var_d['cataclysmic']+var_d['eclipsing']+var_d['xray']+var_d['other'],
         'RR': ['CEP']
         }
    parents = list(d.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(d[parent])

    return d, parents, subclasses


def merge_otype(otype_list):

    # >> merge classes
    parent_dict, parents, subclasses = make_parent_dict()

    new_otype_list = []
    for otype in otype_list:
        if otype in subclasses:
            # >> find parent
            for parent in parents:
                if otype in parent_dict[parent]:
                    new_otype = parent
            new_otype_list.append(new_otype)
        else:
            new_otype_list.append(otype)
    otype_list = new_otype_list

    # >> remove redundant classes 
    redundant_dict, parents, subclasses = make_redundant_otype_dict()
    new_otype_list = []
    for otype in otype_list:
        if otype in parents:
            if len(np.intersect1d(redundant_dict[otype], otype_list))>0:
                new_otype_list.append('')
            else:
                new_otype_list.append(otype)
        else:
            new_otype_list.append(otype)    

            
    otype_list = np.unique(new_otype_list).astype('str')
    otype = np.delete(otype, np.where(otype == ''))
    return otype_list


def get_parent_otypes(ticid, otypes, remove_classes=['PM','IR','UV','X']):
    '''Finds all the objects with same parent and combines them into the same
    class
    '''

    parent_dict = make_parent_dict()
    parents = list(parent_dict.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(parent_dict[parent])

    new_otypes = []
    for i in range(len(otypes)):
        otype = otypes[i].split('|')

        new_otype=[]
        for o in otype:
            if not o in remove_classes:
                if o in subclasses:
                    for parent in parents: # >> find parent otype
                        if o in parent_dict[parent]:
                            new_o = parent
                    new_otype.append(new_o)
                else:
                    new_otype.append(o)

        # >> remove repeats
        new_otype = np.unique(new_otype)

        # >> remove parent if child in otype list e.g. E|EA or E|EW is redundant
        for parent in parents:
            if parent in new_otype:
                if len(np.intersect1d(new_otype, parent_dict[parent]))>0:
                    new_otype = np.delete(new_otype,
                                          np.nonzero(new_otype==parent))

        new_otypes.append('|'.join(new_otype.astype('str')))
    
    # >> get rid of empty classes
    new_otypes = np.array(new_otypes)
    inds = np.nonzero(new_otypes == '')
    new_otypes = np.delete(new_otypes, inds)
    ticid = np.delete(ticid, inds)
                
    return ticid, new_otypes



def get_parents_only(class_info, parent_dict=None,
                     remove_classes=[], remove_flags=[]):
    '''Finds all the objects with same parent and combines them into the same
    class
    TODO: get rid of this function
    '''
    classes = []
    new_class_info = []

    if type(parent_dict) == type(None):
        parent_dict = make_parent_dict()

    parents = list(parent_dict.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(parent_dict[parent])

    for i in range(len(class_info)):
        otype_list = class_info[i][1]

        # >> remove any flags
        for flag in remove_flags:
            otype_list = otype_list.replace(flag, '|')
        otype_list = otype_list.split('|')

        new_otype_list=[]
        for otype in otype_list:
            if not otype in remove_classes:
                if otype in subclasses:
                    # >> find parent
                    for parent in parents:
                        if otype in parent_dict[parent]:
                            new_otype = parent

                    new_otype_list.append(new_otype)
                else:
                    new_otype_list.append(otype)

        # >> remove repeats
        new_otype_list = np.unique(new_otype_list)

        # >> don't want e.g. E|EA or E|EW (redundant)
        if 'E' in new_otype_list:
            if len(np.intersect1d(new_otype_list, ['EA', 'EP', 'EW', 'EB']))>0:
                new_otype_list = np.delete(new_otype_list, np.nonzero(new_otype_list=='E'))

        if 'L' in new_otype_list and len(new_otype_list) > 1:
            new_otype_list = np.delete(new_otype_list,
                                       np.nonzero(new_otype_list=='L'))

        if '' in new_otype_list:
            new_otype_list = np.delete(new_otype_list,
                                       np.nonzero(new_otype_list==''))

        # if '|'.join(new_otype_list) == '|AR|EA|EB|RS|SB':
        #     pdb.set_trace()

        new_class_info.append([class_info[i][0], '|'.join(new_otype_list),
                               class_info[i][2]])

    # >> get rid of empty classes
    new_class_info = np.array(new_class_info)
    new_class_info = np.delete(new_class_info,
                               np.nonzero(new_class_info[:,1]==''), 0)
            
    
    return new_class_info

def make_remove_class_list(simbad=False, rmv_flagged=True):
    '''Currently, our pipeline can only do clustering based on photometric data.
    So classes that require spectroscopic data, etc. are removed.'''
    rmv = ['PM', 'IR', 'nan', 'V', 'VAR', 'As', 'SB', 'LM', 'blu', 'EmO', 'S',
           ]
    sequence_descriptors = ['AB', 'HS', 'BS', 'YSO', 'Y', 'sg', 'BD', 's*b']

    if simbad:
        rmv.append('UV')

    if rmv_flagged:
        flagged = make_flagged_class_list()
    else:
        flagged = []

    return rmv+sequence_descriptors+flagged

def make_flagged_class_list():

    # >> section 5bc of GCVS classifications: eclipsing systems classified
    # >> based on physical characteristics rather than shape of light curve
    eclip = ['GS', 'PN', 'RS', 'WD', 'WR', 'AR', 'D', 'DM', 'DS', 'DW', 'K',
             'KE', 'KW', 'SD'] 

    flagged = ['Em', 'Pe']

    return eclip+flagged

def make_true_label_txt(data_dir, sector):
    '''Combine Sector*_simbad.txt, Sector*_GCVS.txt, and Sector*_asassn.txt
    TODO: edit to handle 30-min cadence, etc.'''
    prefix = data_dir+'databases/Sector'+str(sector)+'_'
    ticid = np.loadtxt(data_dir+'Sector'+str(sector)+'/all_targets_S%03d'%sector\
                       +'_v1.txt')[:,0]
    otypes = {key: [] for key in ticid} # >> initialize

    otypes = read_otype_txt(otypes, prefix+'gcvs.txt', data_dir)
    otypes = read_otype_txt(otypes, prefix+'asassn.txt', data_dir)
    otypes = read_otype_txt(otypes, prefix+'simbad.txt', data_dir, simbad=True)


    pdb.set_trace()
    # >> save to text file
    out = prefix+'true_labels.txt'
    with open(out, 'w') as f:
        for i in range(len(ticid)):
            # >> merge classes
            otype = merge_otype(otypes[ticid[i]])
            otype = '|'.join(otype)
            f.write(str(int(ticid[i]))+','+otype+'\n')    
            
def read_otype_txt(otypes, otype_txt, data_dir, simbad=False, add_chars=['+', '/'],
                   uncertainty_flags=[':', '?', '*']):

    rmv_classes = make_remove_class_list(simbad=simbad)

    if simbad:
        with open(data_dir+'simbad_gcvs_label.txt', 'r') as f:
            lines = f.readlines()
        otype_dict = {}
        for line in lines:
            otype, otype_gcvs = line.split(' = ')
            otype_gcvs = otype_gcvs.replace('\n', '')
            otype_dict[otype] = otype_gcvs

    with open(otype_txt, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            tic, otype, main_id = lines[i].split(',')

            # >> make list of labels
            for char in add_chars:
                otype = otype.replace(char, '|')
            otype_list = otype.split('|')

            # >> remove unceratinty flags
            otype_list_new = []
            stop=False
            for o in otype_list:
                if o == 'UGSU':
                    stop=True
                if len(o) > 0 and o != '**':
                    # >> remove uncertainty flags
                    if o[-1] in uncertainty_flags:
                        o = o[:-1]
                    if '(' in o: # >> remove (B) flag
                        o = o[:o.index('(')]

                    # >> convert to GCVS nomenclature
                    if simbad and o in list(otype_dict.keys()): 
                        o = otype_dict[o]

                    # >> remove classes that require external information
                    if o in rmv_classes:
                        o = ''
                    otype_list_new.append(o)

            otypes[float(tic)] = np.unique(otype_list_new)

    return otypes





def correct_simbad_to_vizier(in_f='./SectorX_simbad.txt',
                             out_f='./SectorX_simbad_revised.txt',
                             simbad_gcvs_conversion='./simbad_gcvs_label.txt',
                             uncertainty_flags=[':', '?', '*']):
    '''TODO: Clean up args.'''
    
    with open(simbad_gcvs_conversion, 'r') as f:
        lines = f.readlines()
    renamed = {}
    for line in lines:
        otype, description = line.split(' = ')
        
        # >> remove new line character
        description = description.replace('\n', '')
        
        renamed[otype] = description    
    
    with open(in_f, 'r') as f:
        lines = f.readlines()
        
        
    for line in lines:
        tic, otype, main = line.split(',')
        otype = otype.replace('+', '|')
        otype_list = otype.split('|')
        otype_list_new = []
        
        for o in otype_list:
            
            if len(o) > 0:
                # >> remove uncertainty_flags
                if o[-1] in uncertainty_flags:
                    o = o[:-1]
                    
                # >> remove (B)
                if '(' in o:
                    o = o[:o.index('(')]
                    
                if o in list(renamed.keys()):
                    o = renamed[o]
                
            otype_list_new.append(o)
                
                
        otype = '|'.join(otype_list_new)
        
        
        with open(out_f, 'a') as f:
            f.write(','.join([tic, otype, main]))

def quick_simbad(ticidasstring):
    """ only returns if it has a tyc id"""
    catalogdata = Catalogs.query_object(ticidasstring, radius=0.02, catalog="TIC")[0]
    try: 
        tyc = "TYC " + catalogdata["TYC"]
        customSimbad = Simbad()
        customSimbad.add_votable_fields("otypes")
        res = customSimbad.query_object(tyc)
        objecttype = res['OTYPES'][0].decode('utf-8')
    except: 
        objecttype = "there is no TYC for this object"
    return objecttype



def get_true_classifications(ticid=[], data_dir='./', sector='all'):
    '''Reads Sector*_true_labels.txt, generated from make_true_label_txt()
    * 
    * sector: either 'all' or int'''
    ticid_true = []
    otypes = []
    database_dir = data_dir+'databases/'
    
    # >> find all text files in directory
    if sector == 'all':
        fnames = fm.filter(os.listdir(database_dir), '*_true_labels.txt')
    else:
        fnames = ['Sector'+str(sector)+'_true_labels.txt']
    
    for fname in fnames:
        data = np.loadtxt(database_dir+fname, delimiter=',', dtype='str')
        ticid_true.extend(data[:,0].astype('float'))
        otypes.extend(data[:,1])

    # >> only return classified targets in ticid list, if given
    if len(ticid) > 0:
        _, inds, _ = np.intersect1d(ticid_true, ticid, return_indices=True)
        ticid_true = np.array(ticid_true)[inds]
        otypes = np.array(otypes)[inds]
    
    return ticid_true, otypes









