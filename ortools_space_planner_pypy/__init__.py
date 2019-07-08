from cppyy.gbl import operations_research
import cppyy

def IntVar(self, *args) -> "operations_research::IntVar *":
    return self.MakeIntVar.__overload__("int64,int64,const string&")(self, *args)

def IsLessVar(self, v, w) -> "operations_research::IntVar *":
    return self.MakeIsLessVar.__overload__("operations_research::IntExpr*const,operations_research::IntExpr*const")(self, v.Var(), w.Var())

def Sum(self, vars: 'std::vector< operations_research::IntVar * > const &') -> "operations_research::IntExpr *":
    vars_list = list(vars)
    v = cppyy.gbl.std.vector["operations_research::IntVar*"]()
    v += [obj.Var() for obj in vars_list]
    return self.MakeSum(v)
    
def Add(self, ct):
    self.AddConstraint(ct)

def Phase(self, *args):
    v = cppyy.gbl.std.vector["operations_research::IntVar*"]()
    v += [obj for obj in args[0]]
    return self.MakePhase(v, args[1], args[2])

def TimeLimit(self, v):
    return self.MakeTimeLimit(v)

def Failures(self):
    return self.failures()

def Branches(self):
    return self.branches()

def Max(self, *args):
    v = cppyy.gbl.std.vector["operations_research::IntVar*"]()
    v += [self.MakeIntVar(0, obj, "") if type(obj) is int else obj.Var() for obj in [*args]]
    return self.MakeMax(v)

def Min(self, *args):
    v = cppyy.gbl.std.vector["operations_research::IntVar*"]()
    v += [obj.Var() for obj in [*args]]
    return self.MakeMin(v)

def __eq__(self, v) -> "operations_research::Constraint *":
    if type(v) is int:
        return self.solver().MakeEquality.__overload__("operations_research::IntExpr*const,int")(self.solver(), self, v)
    else:
        return self.solver().MakeEquality.__overload__("operations_research::IntExpr*const,operations_research::IntExpr*const")(self.solver(), self, v)

def __radd__IntExpr_int(self, v) -> "int64":
    return self.solver().MakeSum.__overload__("operations_research::IntExpr*const,int64")(self.solver(), self.Var(), v)

def __add__IntExpr_IntExpr(self, args) -> "operations_research::IntExpr *":
    return self.solver().MakeSum.__overload__("operations_research::IntExpr*const,operations_research::IntExpr*const")(self.solver(), self.Var(), args.Var())

def __sub__IntExpr_int(self, v) -> "operations_research::IntExpr *":
    if type(v) is int:
        prototype = 'operations_research::IntExpr*const,int64'
    else:
        prototype =  'operations_research::IntExpr*const,operations_research::IntExpr*const'
        v = v.Var()
    return self.solver().MakeSum.__overload__(prototype)(self.solver(), self.Var(), -v)

def __str__Constraint(self) -> "str":
    return str(self.DebugString())

def __mul__(self, v) -> "operations_research::IntExpr *":
    if type(v) is int:
        prototype = 'operations_research::IntExpr*const,int64'
    else:
        prototype =  'operations_research::IntExpr*const,operations_research::IntExpr*const'
        v = v.Var()

    s = self.solver().MakeProd.__overload__(prototype)(self.solver(), self.Var(), v)
    return s

def __radd__Constraint_IntExpr(self, v) -> "operations_research::IntExpr *":
    if type(v) is int:
        prototype = 'operations_research::IntExpr*const,int64'
    else:
        prototype =  'operations_research::IntExpr*const,operations_research::IntExpr*const'
        v = v.Var()

    s = self.solver().MakeSum.__overload__(prototype)(self.solver(), self.Var(), v)
    return s

def __rmul__int_Constraint(self, v) -> "operations_research::IntExpr *":
    if type(v) is int:
        prototype = 'operations_research::IntExpr*const,int64'
    else:
        prototype =  'operations_research::IntExpr*const,operations_research::IntExpr*const'

    return self.solver().MakeProd.__overload__(prototype)(self.solver(), self.Var(), v)

def __ge__IntExpr_IntExpr(self, v) -> "operations_research::IntExpr *":
    if type(v) is int:
        prototype = 'operations_research::IntExpr*const,int64'
    else:
        prototype =  'operations_research::IntExpr*const,operations_research::IntExpr*const'
        v = v.Var()
    return self.solver().MakeGreaterOrEqual.__overload__(prototype)(self.solver(), self.Var(), v)

def __le__IntExpr_int(self, v) -> "operations_research::IntExpr *":
    if type(v) is int:
        prototype = 'operations_research::IntExpr*const,int64'
    else:
        prototype =  'operations_research::IntExpr*const,operations_research::IntExpr*const'
        v = v.Var()
    return self.solver().MakeLessOrEqual.__overload__(prototype)(self.solver(), self.Var(), v)

def __neg__IntExpr(self) -> "operations_research::IntExpr *":
    return self.solver().MakeOpposite.__overload__("operations_research::IntExpr*const")(self.solver(), self.Var())

def __ne__IntExpr__int(self, v) -> "operations_research::IntExpr *":
    if type(v) is int:
        return self.solver().MakeNonEquality.__overload__("operations_research::IntExpr*const,int")(self.solver(),                                                                                     self, v)
    else:
        return self.solver().MakeNonEquality.__overload__(
            "operations_research::IntExpr*const,operations_research::IntExpr*const")(self.solver(), self, v)

def __eq__Constraint_int(self, v) -> "operations_research::Constraint *":
    return self.solver().MakeEquality.__overload__("operations_research::IntExpr*const,int")(self.solver(), self.Var(), v)

def __mul__IntExpr_int(self, v) -> "int64":
    if type(v) is int:
        prototype = 'operations_research::IntExpr*const,int64'
    else:
        prototype = 'operations_research::IntExpr*const,operations_research::IntExpr*const'
        v = v.Var()

    return self.solver().MakeProd.__overload__(prototype)(self.solver(), self.Var(), v)


setattr(operations_research.Solver, "IntVar", IntVar)
setattr(operations_research.Solver, "Phase", Phase)

setattr(operations_research.Solver, "IsLessVar", IsLessVar)
setattr(operations_research.Solver, "Sum", Sum)
setattr(operations_research.Solver, "Add", Add)
setattr(operations_research.Solver, "Max", Max)
setattr(operations_research.Solver, "Min", Min)
setattr(operations_research.Solver, "TimeLimit", TimeLimit)
setattr(operations_research.Solver, "Failures", Failures)
setattr(operations_research.Solver, "Branches", Branches)
setattr(operations_research.IntExpr, "__eq__", __eq__)
setattr(operations_research.IntExpr, "__radd__", __radd__IntExpr_int)
setattr(operations_research.IntExpr, "__add__", __add__IntExpr_IntExpr)
setattr(operations_research.IntExpr, "__mul__", __mul__IntExpr_int)
setattr(operations_research.IntExpr, "__rmul__", __mul__IntExpr_int)
setattr(operations_research.IntExpr, "__sub__", __sub__IntExpr_int)
setattr(operations_research.IntExpr, "__ge__", __ge__IntExpr_IntExpr)
setattr(operations_research.IntExpr, "__le__", __le__IntExpr_int)
setattr(operations_research.IntExpr, "__neg__", __neg__IntExpr)
setattr(operations_research.IntExpr, "__ne__", __ne__IntExpr__int)
setattr(operations_research.Constraint, "__mul__", __mul__)
setattr(operations_research.Constraint, "__radd__", __radd__Constraint_IntExpr)
setattr(operations_research.Constraint, "__rmul__", __rmul__int_Constraint)
setattr(operations_research.Constraint, "__str__", __str__Constraint)
setattr(operations_research.Constraint, "__eq__", __eq__Constraint_int)