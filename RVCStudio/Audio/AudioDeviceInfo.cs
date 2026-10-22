namespace RVCStudio.Audio;
public sealed record AudioDeviceInfo(int Id, string Name)
{
    public override string ToString() => Name;
}
